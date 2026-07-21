"""LangGraph Agent HTTP Server - 用于 K8s 部署"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import HTTPServer, BaseHTTPRequestHandler

from langgraph_agent.conditional_graph import build_graph

# 构建 agent（内部调用 create_chat_model, llm_provider 由环境变量控制）
graph = build_graph()


class AgentHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""

    def _get_trace(self, thread_id: str) -> list:
        """从 state 中获取 trace（简化版，每个请求独立）"""
        return []

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "service": "langgraph-agent"})
        elif self.path == "/":
            self._respond(200, {
                "service": "langgraph-agent",
                "endpoints": {
                    "GET /health": "健康检查",
                    "POST /chat": "发送消息",
                }
            })
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body) if body else {}
            message = data.get("message", "")
            thread_id = data.get("thread_id", "default")

            config = {"configurable": {"thread_id": thread_id}}
            try:
                result = graph.invoke(
                    {"input": message, "trace": [], "conversation_history": []},
                    config=config,
                )
                reply = result.get("output", "（无输出）")
                self._respond(200, {"reply": reply})
            except Exception as e:
                self._respond(500, {"error": str(e), "type": type(e).__name__})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        pass


def main():
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), AgentHandler)
    print(f"[server] LangGraph Agent HTTP server listening on port {port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
