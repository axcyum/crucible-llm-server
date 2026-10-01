import Foundation
import Network

/// Minimal HTTP server that serves an OpenAI-compatible chat completions API
class HTTPServer {
    private var listener: NWListener?
    private let port: UInt16
    private weak var llamaState: LlamaState?

    var isRunning: Bool { listener != nil }

    init(port: UInt16 = 8080) {
        self.port = port
    }

    func start(llamaState: LlamaState) throws {
        self.llamaState = llamaState
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
        listener?.newConnectionHandler = { [weak self] conn in
            self?.handleConnection(conn)
        }
        listener?.start(queue: .global(qos: .userInitiated))
        print("HTTP Server started on port \(port)")
    }

    func stop() {
        listener?.cancel()
        listener = nil
        print("HTTP Server stopped")
    }

    func getLocalIP() -> String {
        var address = "127.0.0.1"
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        if getifaddrs(&ifaddr) == 0 {
            var ptr = ifaddr
            while ptr != nil {
                defer { ptr = ptr?.pointee.ifa_next }
                guard let interface = ptr?.pointee else { continue }
                let addrFamily = interface.ifa_addr.pointee.sa_family
                if addrFamily == UInt8(AF_INET) {
                    let name = String(cString: interface.ifa_name)
                    if name == "en0" || name == "en1" {
                        var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                        getnameinfo(interface.ifa_addr, socklen_t(interface.ifa_addr.pointee.sa_len),
                                    &hostname, socklen_t(hostname.count), nil, 0, NI_NUMERICHOST)
                        address = String(cString: hostname)
                    }
                }
            }
            freeifaddrs(ifaddr)
        }
        return address
    }

    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: .global(qos: .userInitiated))
        receiveRequest(connection, buffered: Data())
    }

    private func receiveRequest(_ connection: NWConnection, buffered: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, _, error in
            guard let self = self, let data = data, error == nil else {
                connection.cancel()
                return
            }
            var bytes = buffered
            bytes.append(data)
            guard bytes.count <= 4 * 1024 * 1024 else {
                self.sendResponse(connection: connection, status: "413 Payload Too Large", body: "{\"error\":\"Request exceeds 4 MiB\"}")
                return
            }
            guard let separator = bytes.range(of: Data("\r\n\r\n".utf8)) else {
                self.receiveRequest(connection, buffered: bytes)
                return
            }
            let header = String(data: bytes[..<separator.lowerBound], encoding: .utf8) ?? ""
            var contentLength = 0
            for line in header.components(separatedBy: "\r\n").dropFirst() {
                let parts = line.split(separator: ":", maxSplits: 1)
                if parts.count == 2 && parts[0].lowercased() == "transfer-encoding" {
                    self.sendResponse(connection: connection, status: "400 Bad Request", body: "{\"error\":\"Use Content-Length; chunked requests are unsupported\"}")
                    return
                }
                if parts.count == 2 && parts[0].lowercased() == "content-length" {
                    guard let length = Int(parts[1].trimmingCharacters(in: .whitespaces)), length >= 0, length <= 4 * 1024 * 1024 else {
                        self.sendResponse(connection: connection, status: "400 Bad Request", body: "{\"error\":\"Invalid Content-Length\"}")
                        return
                    }
                    contentLength = length
                }
            }
            let end = separator.upperBound + contentLength
            if bytes.count < end {
                self.receiveRequest(connection, buffered: bytes)
                return
            }
            guard let request = String(data: bytes[..<end], encoding: .utf8) else {
                self.sendResponse(connection: connection, status: "400 Bad Request", body: "{\"error\":\"Invalid UTF-8\"}")
                return
            }
            self.routeRequest(request, connection: connection)
        }
    }

    private func routeRequest(_ request: String, connection: NWConnection) {
        let lines = request.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else {
            sendResponse(connection: connection, status: "400 Bad Request", body: "{\"error\":\"Bad Request\"}")
            return
        }

        let parts = requestLine.components(separatedBy: " ")
        guard parts.count >= 2 else {
            sendResponse(connection: connection, status: "400 Bad Request", body: "{\"error\":\"Bad Request\"}")
            return
        }

        let method = parts[0]
        let path = parts[1]

        // CORS headers for all responses
        if method == "OPTIONS" {
            sendResponse(connection: connection, status: "200 OK", body: "", extraHeaders: [
                "Access-Control-Allow-Origin: *",
                "Access-Control-Allow-Methods: GET, POST, OPTIONS",
                "Access-Control-Allow-Headers: Content-Type, Authorization"
            ])
            return
        }

        switch path {
        case "/":
            sendResponse(connection: connection, status: "200 OK", body: "Crucible LLM Server is running")
        case "/v1/models", "/api/tags":
            handleModels(connection: connection)
        case "/v1/chat/completions":
            if method == "POST" {
                handleChatCompletion(request: request, connection: connection)
            } else {
                sendResponse(connection: connection, status: "405 Method Not Allowed", body: "{\"error\":\"Method Not Allowed\"}")
            }
        default:
            sendResponse(connection: connection, status: "404 Not Found", body: "{\"error\":\"Not Found\"}")
        }
    }

    private func handleModels(connection: NWConnection) {
        let response = """
        {"object":"list","data":[{"id":"local","object":"model","created":0,"owned_by":"local"}]}
        """
        sendResponse(connection: connection, status: "200 OK", body: response, contentType: "application/json")
    }

    private func handleChatCompletion(request: String, connection: NWConnection) {
        // Extract JSON body from HTTP request
        guard let bodyStart = request.range(of: "\r\n\r\n")?.upperBound else {
            sendResponse(connection: connection, status: "400 Bad Request", body: "{\"error\":\"No body\"}", contentType: "application/json")
            return
        }

        let bodyString = String(request[bodyStart...])
        guard let bodyData = bodyString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any],
              let messages = json["messages"] as? [[String: Any]] else {
            sendResponse(connection: connection, status: "400 Bad Request", body: "{\"error\":\"Invalid JSON\"}", contentType: "application/json")
            return
        }

        guard !messages.isEmpty else {
            sendResponse(connection: connection, status: "400 Bad Request", body: "{\"error\":\"messages must not be empty\"}")
            return
        }
        for message in messages {
            if let blocks = message["content"] as? [[String: Any]], blocks.contains(where: { ($0["type"] as? String) != "text" }) {
                sendResponse(connection: connection, status: "400 Bad Request", body: "{\"error\":\"Only text content is supported\"}")
                return
            }
        }
        let maxTokens = max(1, min(json["max_completion_tokens"] as? Int ?? json["max_tokens"] as? Int ?? 512, 2048))

        // Run inference on a background thread
        Task {
            guard let llamaState = await self.llamaState else {
                self.sendResponse(connection: connection, status: "500 Internal Server Error",
                                  body: "{\"error\":\"No model loaded\"}", contentType: "application/json")
                return
            }

            let result: String
            do {
                let prompt = OpenAIChat.prompt(json, gemma: await llamaState.usesGemma)
                result = try await llamaState.completeForAPI(text: prompt, maxTokens: maxTokens)
            } catch {
                let status: String
                let message: String
                switch error {
                case LlamaError.contextOverflow:
                    status = "400 Bad Request"
                    message = "Prompt exceeds the 8192-token context. Shorten history or reduce tools."
                case APIInferenceError.busy:
                    status = "503 Service Unavailable"
                    message = "Model is busy. Send one request at a time."
                case APIInferenceError.noModel:
                    status = "503 Service Unavailable"
                    message = "Load a model in the iPad app first."
                default:
                    status = "500 Internal Server Error"
                    message = "Inference failed."
                }
                self.sendResponse(connection: connection, status: status,
                                  body: OpenAIChat.json(["error": ["message": message, "type": "invalid_request_error"]]), contentType: "application/json")
                return
            }
            let message = OpenAIChat.message(result, request: json)
            let id = "chatcmpl-\(UUID().uuidString)"
            let created = Int(Date().timeIntervalSince1970)
            if json["stream"] as? Bool == true {
                self.sendResponse(connection: connection, status: "200 OK",
                                  body: OpenAIChat.stream(message: message, id: id, created: created, model: "local"),
                                  contentType: "text/event-stream", extraHeaders: ["Cache-Control: no-cache"])
                return
            }

            let responseJSON: [String: Any] = [
                "id": id,
                "object": "chat.completion",
                "created": created,
                "model": "local",
                "choices": [[
                    "index": 0,
                    "message": message,
                    "finish_reason": message["tool_calls"] == nil ? "stop" : "tool_calls"
                ]]
            ]

            if let jsonData = try? JSONSerialization.data(withJSONObject: responseJSON),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                self.sendResponse(connection: connection, status: "200 OK", body: jsonString, contentType: "application/json")
            } else {
                self.sendResponse(connection: connection, status: "500 Internal Server Error",
                                  body: "{\"error\":\"Failed to serialize response\"}", contentType: "application/json")
            }
        }
    }

    private func sendResponse(connection: NWConnection, status: String, body: String,
                              contentType: String = "text/plain", extraHeaders: [String] = []) {
        var headers = "HTTP/1.1 \(status)\r\n"
        headers += "Content-Type: \(contentType)\r\n"
        headers += "Content-Length: \(body.utf8.count)\r\n"
        headers += "Access-Control-Allow-Origin: *\r\n"
        headers += "Connection: close\r\n"
        for header in extraHeaders {
            headers += "\(header)\r\n"
        }
        headers += "\r\n"

        let responseData = (headers + body).data(using: .utf8)!
        connection.send(content: responseData, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
}
