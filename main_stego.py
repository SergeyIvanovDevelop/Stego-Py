import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets
import design_stego  # Это наш конвертированный файл дизайна
from PyQt5.QtGui import QPixmap
import Insert_Data
import Extract_Data
import os






class ExampleApp(QtWidgets.QMainWindow, design_stego.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py

        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна



        # ------- Код выполняющийся при старте приложения --------------------
        #self.setWindowTitle('Тест')
        #self.setStyleSheet("background-color: rgba(66, 66, 99, 255)")
        self.outputFormat.addItems(["png", "bmp", "jp2", "jpg"])
        self.N_levels.addItems(["1", "2", "3", "4", "5"])

        self.getFileNameInsert.clicked.connect(self.GFNI)
        self.getFileNameExtract.clicked.connect(self.GFNE)
        self.getKeyPathInsert.clicked.connect(self.GKFNI)
        self.getKeyPathExtract.clicked.connect(self.GKFNE)

        self.Insert.clicked.connect(self.Insert_f)
        self.Extract.clicked.connect(self.Extract_f)

        #self.outputFormat.setStyleSheet("QComboBox {color:rgba(255, 255, 255, 255)}")
        #self.PathToKeyFileInsert.setStyleSheet("QTextEdit {color:rgba(255, 255, 255, 255)}")
        #self.PathToKeyFileExtract.setStyleSheet("QTextEdit {color:rgba(255, 255, 255, 255)}")
        #self.PathToExtractFile.setStyleSheet("QTextEdit {color:rgba(255, 255, 255, 255)}")
        #self.PathToInsertFile.setStyleSheet("QTextEdit {color:rgba(255, 255, 255, 255)}")
        #self.TextInsert.setStyleSheet("QTextEdit {color:rgba(255, 255, 255, 255)}")
        #self.TextExtract.setStyleSheet("QTextEdit {color:rgba(255, 255, 255, 255)}")
        # ------- Код выполняющийся при старте приложения --------------------


    def GFNI(self):
        filename = self.getFileName()
        self.PathToInsertFile.setText(filename)
        pixmap = QPixmap(filename)
        self.ImageInsert.setPixmap(pixmap)

    def GFNE(self):
        filename = self.getFileName()
        self.PathToExtractFile.setText(filename)
        pixmap = QPixmap(filename)
        self.ImageExtract.setPixmap(pixmap)

    def GKFNI(self):
        filename = self.getFileName()
        self.PathToKeyFileInsert.setText(filename)

    def GKFNE(self):
        filename = self.getFileName()
        self.PathToKeyFileExtract.setText(filename)


    def Insert_f(self):
        #Insert
        secret_message = self.TextInsert.text()
        filename = self.PathToInsertFile.text()
        file_key = self.PathToKeyFileInsert.text()
        output_format = self.outputFormat.currentText()
        n_levels = int(self.N_levels.currentText())
        print("output_format = |" + output_format+"|")
        result = Insert_Data.paste_data_to_image(filename, output_format, file_key, secret_message, n_levels)
        if result != 0:
            self.label.setText("Не удачно")
            self.label.setStyleSheet("QLabel {color:rgba(255, 0, 0, 255)}")
        else:
            self.label.setText("Удачно")
            self.label.setStyleSheet("QLabel {color:rgba(0, 255, 0, 255)}")


    def Extract_f(self):
        #Extract
        filename = self.PathToExtractFile.text()
        file_key = self.PathToKeyFileExtract.text()
        n_levels = int(self.N_levels.currentText())
        exctract_message = Extract_Data.extract_data(filename, file_key, n_levels)
        self.TextExtract.setText(exctract_message)

    def getFileName(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл")
        # открыть диалог выбора директории и установить значение переменной
        # равной пути к выбранному файлу

        return filename[0]



def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()