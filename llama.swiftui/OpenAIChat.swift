import Foundation

/// Wire-format adapter. Tool execution stays on the client (OpenClaude).
enum OpenAIChat {
    static func json(_ value: Any) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8) else { return "{}" }
        return text
    }

    static func text(_ content: Any?) -> String {
        if let value = content as? String { return value }
        if let blocks = content as? [[String: Any]] {
            return blocks.compactMap { $0["text"] as? String }.joined(separator: "\n")
        }
        return ""
    }

    static func functions(_ request: [String: Any]) -> [[String: Any]] {
        if request["tool_choice"] as? String == "none" { return [] }
        return (request["tools"] as? [[String: Any]] ?? []).compactMap { $0["function"] as? [String: Any] }
    }

    static func prompt(_ request: [String: Any], gemma: Bool = false) -> String {
        let functions = functions(request)
        var instruction = (gemma ? "" : "/no_think\n") + "You are a helpful coding assistant."
        if !functions.isEmpty {
            instruction += """
            \nYou may call these functions: \(json(functions))
            To use a function, respond with ONLY <tool_call>{"name":"function_name","arguments":{"parameter":"value"}}</tool_call>.
            Use exact function names and valid JSON arguments matching the schema. Do not pretend to execute tools yourself. Tool results will follow in a tool message. Otherwise answer normally.
            """
            if request["tool_choice"] as? String == "required" { instruction += "\nYou must call a function." }
            if let choice = request["tool_choice"] as? [String: Any],
               let function = choice["function"] as? [String: Any], let name = function["name"] as? String {
                instruction += "\nCall the function \(name)."
            }
        }
        var turns: [(role: String, content: String)] = [("system", instruction)]
        for message in request["messages"] as? [[String: Any]] ?? [] {
            let role = message["role"] as? String ?? "user"
            var content = text(message["content"])
            if let calls = message["tool_calls"] as? [[String: Any]] {
                for call in calls {
                    guard let function = call["function"] as? [String: Any] else { continue }
                    let rawArguments = function["arguments"] as? String ?? "{}"
                    let arguments = rawArguments.data(using: .utf8).flatMap { try? JSONSerialization.jsonObject(with: $0) } ?? [:]
                    content += "\n<tool_call>\(json(["name": function["name"] ?? "", "arguments": arguments]))</tool_call>"
                }
            }
            if role == "tool", let id = message["tool_call_id"] as? String { content = "Tool result (\(id)):\n" + content }
            turns.append((role, content))
        }
        if gemma {
            let system = turns.filter { $0.role == "system" || $0.role == "developer" }.map { $0.content }.joined(separator: "\n\n")
            var merged: [(role: String, content: String)] = []
            for turn in turns where turn.role != "system" && turn.role != "developer" {
                let role = turn.role == "assistant" ? "model" : "user"
                var content = turn.content
                if merged.isEmpty { content = system + "\n\n" + content }
                if merged.last?.role == role {
                    merged[merged.count - 1].content += "\n\n" + content
                } else {
                    merged.append((role, content))
                }
            }
            return "<bos>" + merged.map { "<start_of_turn>\($0.role)\n\($0.content)<end_of_turn>\n" }.joined() + "<start_of_turn>model\n"
        }
        return turns.map { "<|im_start|>\($0.role)\n\($0.content)<|im_end|>\n" }.joined() + "<|im_start|>assistant\n"
    }

    static func message(_ output: String, request: [String: Any]) -> [String: Any] {
        let allowed = Set(functions(request).compactMap { $0["name"] as? String })
        guard !allowed.isEmpty else { return ["role": "assistant", "content": output] }
        var candidates: [String] = []
        let pattern = "<tool_call>\\s*([\\s\\S]*?)\\s*</tool_call>"
        if let regex = try? NSRegularExpression(pattern: pattern) {
            for match in regex.matches(in: output, range: NSRange(output.startIndex..., in: output)) {
                if let range = Range(match.range(at: 1), in: output) { candidates.append(String(output[range])) }
            }
        }
        if candidates.isEmpty {
            var bare = output.trimmingCharacters(in: .whitespacesAndNewlines)
            if bare.hasPrefix("```"), let newline = bare.firstIndex(of: "\n"), bare.hasSuffix("```") {
                bare = String(bare[bare.index(after: newline)..<bare.index(bare.endIndex, offsetBy: -3)])
            }
            candidates = [bare]
        }
        var calls: [[String: Any]] = []
        for candidate in candidates {
            guard let data = candidate.data(using: .utf8),
                  let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = value["name"] as? String, allowed.contains(name) else { continue }
            var arguments = value["arguments"]
            if let raw = arguments as? String, let data = raw.data(using: .utf8) {
                arguments = try? JSONSerialization.jsonObject(with: data)
            }
            guard let arguments = arguments as? [String: Any] else { continue }
            calls.append(["id": "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
                          "type": "function", "function": ["name": name, "arguments": json(arguments)]])
        }
        if calls.isEmpty { return ["role": "assistant", "content": output] }
        return ["role": "assistant", "content": NSNull(), "tool_calls": calls]
    }

    /// Buffered SSE: compatible with streaming clients; delivered after inference finishes.
    static func stream(message: [String: Any], id: String, created: Int, model: String) -> String {
        func event(_ delta: [String: Any], finish: Any = NSNull()) -> String {
            "data: " + json(["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                             "choices": [["index": 0, "delta": delta, "finish_reason": finish]]]) + "\n\n"
        }
        var body = event(["role": "assistant", "content": ""])
        if let calls = message["tool_calls"] as? [[String: Any]] {
            body += event(["tool_calls": calls.enumerated().map { index, call -> [String: Any] in
                var delta = call
                delta["index"] = index
                return delta
            }])
            body += event([:], finish: "tool_calls")
        } else {
            body += event(["content": message["content"] as? String ?? ""])
            body += event([:], finish: "stop")
        }
        return body + "data: [DONE]\n\n"
    }
}
