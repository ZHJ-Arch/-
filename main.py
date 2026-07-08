import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QPushButton, QTextEdit


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Legrest Linkage Optimizer")
        self.resize(1000, 700)

        central = QWidget()
        layout = QVBoxLayout(central)

        title = QLabel("Legrest Linkage Optimizer")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        info = QTextEdit()
        info.setReadOnly(True)
        info.setText(
            "腿托连杆优化软件原型已成功启动。\\n\\n"
            "当前版本用于验证 Windows EXE 打包流程。\\n"
            "后续可继续接入完整连杆优化计算界面。\\n\\n"
            "如果你能看到这个窗口，说明 PySide6 桌面程序已经打包成功。"
        )

        button = QPushButton("运行测试")
        button.clicked.connect(lambda: info.append("\\n测试按钮已点击，软件运行正常。"))

        layout.addWidget(title)
        layout.addWidget(info)
        layout.addWidget(button)

        self.setCentralWidget(central)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
