import Foundation

@main
struct OpenAIChatTests {
    static func main() {
        let request: [String: Any] = [
            "messages": [["role": "user", "content": [["type": "text", "text": "Read a file"]]]],
            "tools": [["type": "function", "function": ["name": "Read", "parameters": ["type": "object"]]]]
        ]
        precondition(OpenAIChat.prompt(request).contains("Read a file"))
        precondition(OpenAIChat.prompt(request).contains("function_name"))
        let gemma = OpenAIChat.prompt(request, gemma: true)
        precondition(gemma.hasPrefix("<bos><start_of_turn>user\n"))
        precondition(gemma.hasSuffix("<start_of_turn>model\n"))
        precondition(!gemma.contains("<|im_start|>"))
        let message = OpenAIChat.message("<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"C:\\\\test.txt\"}}</tool_call>", request: request)
        let calls = message["tool_calls"] as! [[String: Any]]
        precondition(calls.count == 1)
        let function = calls[0]["function"] as! [String: Any]
        precondition(function["name"] as? String == "Read")
        precondition((function["arguments"] as? String)?.contains("file_path") == true)
        precondition(OpenAIChat.message("<tool_call>{\"name\":\"Unknown\",\"arguments\":{}}</tool_call>", request: request)["tool_calls"] == nil)
        precondition(OpenAIChat.message("<tool_call>broken</tool_call>", request: request)["tool_calls"] == nil)
        var disabled = request
        disabled["tool_choice"] = "none"
        precondition(OpenAIChat.message("{\"name\":\"Read\",\"arguments\":{}}", request: disabled)["tool_calls"] == nil)
        let stream = OpenAIChat.stream(message: message, id: "test", created: 1, model: "local")
        precondition(stream.hasSuffix("data: [DONE]\n\n"))
        for line in stream.components(separatedBy: "\n") where line.hasPrefix("data: {") {
            let chunk = try! JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as! [String: Any]
            precondition(chunk["object"] as? String == "chat.completion.chunk")
        }
        precondition(stream.contains("tool_calls"))
        let normal = OpenAIChat.stream(message: ["role": "assistant", "content": "日本語"], id: "test", created: 1, model: "local")
        precondition(normal.contains("日本語"))
        print("OpenAI chat adapter tests passed")
    }
}
