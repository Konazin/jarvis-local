import pickle
import socket
import sys


def read_request(connection: socket.socket) -> dict:
    size = int.from_bytes(connection.recv(8), "big")
    data = b""
    while len(data) < size:
        data += connection.recv(size - len(data))
    return pickle.loads(data)


def send_response(connection: socket.socket, payload: dict) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(len(data).to_bytes(8, "big") + data)


def main() -> None:
    path, lang_code, voice, speed = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    from kokoro import KPipeline

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    connection, _ = server.accept()
    pipeline = None
    try:
        while True:
            request = read_request(connection)
            if request["command"] == "LOAD":
                pipeline = KPipeline(lang_code=lang_code)
                send_response(connection, {"status": "ready"})
            elif request["command"] == "SPEAK":
                import numpy as np

                chunks = [audio.numpy() for _, _, audio in pipeline(request["text"], voice=voice, speed=speed)]
                send_response(connection, {"status": "audio", "audio": np.concatenate(chunks)})
            elif request["command"] == "STOP":
                send_response(connection, {"status": "stopped"})
                return
    except (BrokenPipeError, ConnectionError):
        return
    finally:
        connection.close()
        server.close()


if __name__ == "__main__":
    main()
