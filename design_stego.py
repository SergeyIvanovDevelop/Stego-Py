# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design_stego.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(973, 636)
        MainWindow.setMinimumSize(QtCore.QSize(973, 636))
        MainWindow.setMaximumSize(QtCore.QSize(973, 636))
        MainWindow.setStyleSheet("color: rgb(238, 238, 236);\n"
"background-color: rgb(85, 87, 83);\n"
"border-color: rgb(46, 52, 54);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ImageInsert = QtWidgets.QLabel(self.centralwidget)
        self.ImageInsert.setGeometry(QtCore.QRect(60, 260, 341, 221))
        self.ImageInsert.setText("")
        self.ImageInsert.setScaledContents(True)
        self.ImageInsert.setObjectName("ImageInsert")
        self.ImageExtract = QtWidgets.QLabel(self.centralwidget)
        self.ImageExtract.setGeometry(QtCore.QRect(550, 260, 341, 221))
        self.ImageExtract.setStyleSheet("border-color: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(0, 0, 0, 0), stop:0.52 rgba(0, 0, 0, 0), stop:0.565 rgba(82, 121, 76, 33), stop:0.65 rgba(159, 235, 148, 64), stop:0.721925 rgba(255, 238, 150, 129), stop:0.77 rgba(255, 128, 128, 204), stop:0.89 rgba(191, 128, 255, 64), stop:1 rgba(0, 0, 0, 0));")
        self.ImageExtract.setText("")
        self.ImageExtract.setScaledContents(True)
        self.ImageExtract.setObjectName("ImageExtract")
        self.PathToInsertFile = QtWidgets.QLineEdit(self.centralwidget)
        self.PathToInsertFile.setGeometry(QtCore.QRect(50, 80, 191, 25))
        self.PathToInsertFile.setObjectName("PathToInsertFile")
        self.PathToExtractFile = QtWidgets.QLineEdit(self.centralwidget)
        self.PathToExtractFile.setGeometry(QtCore.QRect(540, 80, 191, 25))
        self.PathToExtractFile.setObjectName("PathToExtractFile")
        self.PathToKeyFileInsert = QtWidgets.QLineEdit(self.centralwidget)
        self.PathToKeyFileInsert.setGeometry(QtCore.QRect(50, 140, 191, 25))
        self.PathToKeyFileInsert.setObjectName("PathToKeyFileInsert")
        self.PathToKeyFileExtract = QtWidgets.QLineEdit(self.centralwidget)
        self.PathToKeyFileExtract.setGeometry(QtCore.QRect(540, 140, 191, 25))
        self.PathToKeyFileExtract.setObjectName("PathToKeyFileExtract")
        self.getFileNameInsert = QtWidgets.QPushButton(self.centralwidget)
        self.getFileNameInsert.setGeometry(QtCore.QRect(250, 80, 161, 25))
        self.getFileNameInsert.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.getFileNameInsert.setObjectName("getFileNameInsert")
        self.getKeyPathInsert = QtWidgets.QPushButton(self.centralwidget)
        self.getKeyPathInsert.setGeometry(QtCore.QRect(250, 140, 161, 25))
        self.getKeyPathInsert.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.getKeyPathInsert.setObjectName("getKeyPathInsert")
        self.getFileNameExtract = QtWidgets.QPushButton(self.centralwidget)
        self.getFileNameExtract.setGeometry(QtCore.QRect(740, 80, 161, 25))
        self.getFileNameExtract.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.getFileNameExtract.setObjectName("getFileNameExtract")
        self.getKeyPathExtract = QtWidgets.QPushButton(self.centralwidget)
        self.getKeyPathExtract.setGeometry(QtCore.QRect(740, 140, 161, 25))
        self.getKeyPathExtract.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.getKeyPathExtract.setObjectName("getKeyPathExtract")
        self.TextInsert = QtWidgets.QLineEdit(self.centralwidget)
        self.TextInsert.setGeometry(QtCore.QRect(50, 510, 361, 25))
        self.TextInsert.setObjectName("TextInsert")
        self.TextExtract = QtWidgets.QLineEdit(self.centralwidget)
        self.TextExtract.setGeometry(QtCore.QRect(540, 510, 361, 25))
        self.TextExtract.setObjectName("TextExtract")
        self.Insert = QtWidgets.QPushButton(self.centralwidget)
        self.Insert.setGeometry(QtCore.QRect(110, 540, 231, 25))
        self.Insert.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.Insert.setObjectName("Insert")
        self.Extract = QtWidgets.QPushButton(self.centralwidget)
        self.Extract.setGeometry(QtCore.QRect(590, 540, 271, 25))
        self.Extract.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.Extract.setObjectName("Extract")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(250, 590, 151, 17))
        self.label.setText("")
        self.label.setObjectName("label")
        self.outputFormat = QtWidgets.QComboBox(self.centralwidget)
        self.outputFormat.setGeometry(QtCore.QRect(50, 210, 86, 25))
        self.outputFormat.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.outputFormat.setObjectName("outputFormat")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 190, 191, 17))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(50, 590, 181, 17))
        self.label_3.setObjectName("label_3")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(460, 0, 31, 711))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.line.setFont(font)
        self.line.setStyleSheet("border-color: rgb(238, 238, 236);\n"
"color: rgb(238, 238, 236);\n"
"gridline-color: rgb(238, 238, 236);")
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(140, 20, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(630, 20, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.N_levels = QtWidgets.QComboBox(self.centralwidget)
        self.N_levels.setGeometry(QtCore.QRect(250, 210, 86, 25))
        self.N_levels.setObjectName("N_levels")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(250, 190, 171, 17))
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Steganography"))
        self.getFileNameInsert.setText(_translate("MainWindow", "Путь к изображению"))
        self.getKeyPathInsert.setText(_translate("MainWindow", "Путь к ключу"))
        self.getFileNameExtract.setText(_translate("MainWindow", "Путь к изображению"))
        self.getKeyPathExtract.setText(_translate("MainWindow", "Путь к ключу"))
        self.Insert.setText(_translate("MainWindow", "Внедрить информацию"))
        self.Extract.setText(_translate("MainWindow", "Извлечь информацию"))
        self.label_2.setText(_translate("MainWindow", "Формат выходного файла"))
        self.label_3.setText(_translate("MainWindow", "Результат внедрения:"))
        self.label_4.setText(_translate("MainWindow", "Внедрение данных"))
        self.label_5.setText(_translate("MainWindow", "Извлечение данных"))
        self.label_6.setText(_translate("MainWindow", "Уровней декомпозиции"))