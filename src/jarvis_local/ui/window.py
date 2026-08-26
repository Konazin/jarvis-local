import logging

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QVBoxLayout, QWidget

from jarvis_local.core.assistant import Assistant


class AskWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, assistant: Assistant, text: str) -> None:
        super().__init__()
        self.assistant, self.text = assistant, text

    def run(self) -> None:
        try:
            self.finished.emit(self.assistant.ask(self.text))
        except Exception as exc:
            logging.exception("falha na mensagem")
            self.failed.emit(str(exc))


class Window(QWidget):
    state_changed = Signal(str)

    def __init__(self, assistant: Assistant) -> None:
        super().__init__()

        self.assistant = assistant

        self.setWindowTitle("Yuki")
        self.resize(480, 360)

        self.status = QLabel("IDLE")
        self.history = QListWidget()
        self.input = QLineEdit()

        self.assistant.on_state_change = self.state_changed.emit
        self.state_changed.connect(self.status.setText)

        self.input.setPlaceholderText("Digite uma mensagem...")

        send = QPushButton("Enviar")
        send.clicked.connect(self.ask)
        self.input.returnPressed.connect(self.ask)

        row = QHBoxLayout()
        row.addWidget(self.input)
        row.addWidget(send)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.history)
        layout.addLayout(row)

    def ask(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.input.setEnabled(False)
        self.status.setText("THINKING")
        self.history.addItem(f"Você: {text}")
        self.thread = QThread(self)
        self.worker = AskWorker(self.assistant, text)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.done)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def done(self, answer: str) -> None:
        self.history.addItem(f"Yuki: {answer}")
        self.input.setEnabled(True)

    def failed(self, error: str) -> None:
        self.history.addItem(f"Erro: {error}")
        self.status.setText("IDLE")
        self.input.setEnabled(True)

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide()
