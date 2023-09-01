# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'periodic_input.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(262, 234)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(60, 180, 191, 50))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.freq_SpinBox = QtWidgets.QDoubleSpinBox(Dialog)
        self.freq_SpinBox.setGeometry(QtCore.QRect(160, 10, 91, 41))
        self.freq_SpinBox.setDecimals(1)
        self.freq_SpinBox.setMinimum(1.0)
        self.freq_SpinBox.setObjectName("freq_SpinBox")
        self.amp_spinBox = QtWidgets.QDoubleSpinBox(Dialog)
        self.amp_spinBox.setGeometry(QtCore.QRect(160, 70, 91, 41))
        self.amp_spinBox.setDecimals(1)
        self.amp_spinBox.setMinimum(0.1)
        self.amp_spinBox.setObjectName("amp_spinBox")
        self.freq_width_spinBox = QtWidgets.QDoubleSpinBox(Dialog)
        self.freq_width_spinBox.setGeometry(QtCore.QRect(160, 130, 91, 41))
        self.freq_width_spinBox.setDecimals(1)
        self.freq_width_spinBox.setMinimum(0.1)
        self.freq_width_spinBox.setObjectName("freq_width_spinBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 20, 81, 17))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 80, 91, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(20, 140, 121, 17))
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Input Periodic"))
        self.label.setText(_translate("Dialog", "Frequency"))
        self.label_2.setText(_translate("Dialog", "Amplitude"))
        self.label_3.setText(_translate("Dialog", "Frequency width"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
