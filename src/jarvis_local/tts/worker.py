import json
import os
import socket
import sys


def read_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise ConnectionError("manager encerrou a conexão")
        result.extend(chunk)
    return bytes(result)


def receive(connection: socket.socket) -> tuple[dict, bytes]:
    header_size = int.from_bytes(read_exact(connection, 8), "big")
    header = json.loads(read_exact(connection, header_size))
    return header, read_exact(connection, header.get("payload_size", 0))


def send(connection: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = {**header, "payload_size": len(payload)}
    encoded = json.dumps(header).encode()
    connection.sendall(len(encoded).to_bytes(8, "big") + encoded + payload)


def main() -> None:
    path, lang_code, voice, speed = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o600)
    server.listen(1)
    connection, _ = server.accept()
    pipeline = None
    try:
        while True:
            request, payload = receive(connection)
            command = request.get("command")
            try:
                if command == "LOAD":
                    from kokoro import KPipeline

                    pipeline = KPipeline(lang_code=lang_code)
                    send(connection, {"status": "ready"})
                elif command == "SPEAK":
                    import numpy as np

                    chunks = [audio.numpy() for _, _, audio in pipeline(payload.decode(), voice=voice, speed=speed)]
                    audio = np.concatenate(chunks).astype(np.float32, copy=False)
                    send(
                        connection,
                        {
                            "status": "audio",
                            "sample_rate": 24000,
                            "dtype": "float32",
                            "audio_samples": int(audio.size),
                            "audio_duration_ms": float(audio.size * 1000 / 24000),
                        },
                        audio.tobytes(),
                    )
                elif command == "STOP":
                    send(connection, {"status": "stopped"})
                    return
                else:
                    send(connection, {"status": "error", "error": f"comando desconhecido: {command}"})
            except Exception as exc:
                send(connection, {"status": "error", "error": str(exc)})
    except (BrokenPipeError, ConnectionError, json.JSONDecodeError):
        return
    finally:
        connection.close()
        server.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
