import traceback

import tipc
import torch
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/update_weight", methods=["POST"])
def update_weight():
    try:
        data = request.get_json()
        name = data["name"]
        desc = data["desc"]
        meta = tipc.SharedMemIpcMeta.decode(desc)
        helper = tipc.SharedMemoryIPCHelper()
        tensor = helper.build_from_meta(meta)
        print(name, tensor)
        return {"message": "ok"}
    except Exception as e:
        print(e)
        print(traceback.format_stack())


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8080)
