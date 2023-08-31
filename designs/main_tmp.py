# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main2.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.resize(1532, 912)
        self.centralwidget = QtWidgets.QWidget(dialog)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1531, 891))
        self.tabWidget.setObjectName("tabWidget")
        self.sig_gen_tab = QtWidgets.QWidget()
        self.sig_gen_tab.setAccessibleDescription("")
        self.sig_gen_tab.setAutoFillBackground(False)
        self.sig_gen_tab.setObjectName("sig_gen_tab")
        self.label = QtWidgets.QLabel(self.sig_gen_tab)
        self.label.setGeometry(QtCore.QRect(10, 10, 131, 17))
        self.label.setObjectName("label")
        self.remove_per_button = QtWidgets.QPushButton(self.sig_gen_tab)
        self.remove_per_button.setGeometry(QtCore.QRect(70, 460, 81, 25))
        self.remove_per_button.setObjectName("remove_per_button")
        self.ap_model_comboBox = QtWidgets.QComboBox(self.sig_gen_tab)
        self.ap_model_comboBox.setGeometry(QtCore.QRect(80, 45, 221, 25))
        self.ap_model_comboBox.setObjectName("ap_model_comboBox")
        self.ap_model_comboBox.addItem("")
        self.ap_model_comboBox.addItem("")
        self.add_per_button = QtWidgets.QPushButton(self.sig_gen_tab)
        self.add_per_button.setGeometry(QtCore.QRect(10, 460, 51, 25))
        self.add_per_button.setObjectName("add_per_button")
        self.ap_stackedWidget = QtWidgets.QStackedWidget(self.sig_gen_tab)
        self.ap_stackedWidget.setGeometry(QtCore.QRect(10, 130, 321, 291))
        self.ap_stackedWidget.setObjectName("ap_stackedWidget")
        self.fixed_ap_page = QtWidgets.QWidget()
        self.fixed_ap_page.setObjectName("fixed_ap_page")
        self.label_4 = QtWidgets.QLabel(self.fixed_ap_page)
        self.label_4.setGeometry(QtCore.QRect(20, 70, 67, 20))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.fixed_ap_page)
        self.label_5.setGeometry(QtCore.QRect(20, 30, 71, 17))
        self.label_5.setObjectName("label_5")
        self.ap_fixed_exponent_spinBox = QtWidgets.QDoubleSpinBox(self.fixed_ap_page)
        self.ap_fixed_exponent_spinBox.setGeometry(QtCore.QRect(100, 68, 81, 26))
        self.ap_fixed_exponent_spinBox.setMinimum(-100.0)
        self.ap_fixed_exponent_spinBox.setMaximum(-1.0)
        self.ap_fixed_exponent_spinBox.setSingleStep(0.5)
        self.ap_fixed_exponent_spinBox.setObjectName("ap_fixed_exponent_spinBox")
        self.ap_fixed_intercept_spinBox = QtWidgets.QDoubleSpinBox(self.fixed_ap_page)
        self.ap_fixed_intercept_spinBox.setGeometry(QtCore.QRect(100, 27, 81, 26))
        self.ap_fixed_intercept_spinBox.setMinimum(-99.0)
        self.ap_fixed_intercept_spinBox.setSingleStep(0.5)
        self.ap_fixed_intercept_spinBox.setObjectName("ap_fixed_intercept_spinBox")
        self.ap_stackedWidget.addWidget(self.fixed_ap_page)
        self.knee_ap_page = QtWidgets.QWidget()
        self.knee_ap_page.setObjectName("knee_ap_page")
        self.ap_stackedWidget.addWidget(self.knee_ap_page)
        self.label_3 = QtWidgets.QLabel(self.sig_gen_tab)
        self.label_3.setGeometry(QtCore.QRect(20, 41, 61, 30))
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(self.sig_gen_tab)
        self.label_2.setGeometry(QtCore.QRect(10, 420, 111, 17))
        self.label_2.setObjectName("label_2")
        self.per_table = QtWidgets.QTableWidget(self.sig_gen_tab)
        self.per_table.setGeometry(QtCore.QRect(10, 490, 311, 191))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.per_table.sizePolicy().hasHeightForWidth())
        self.per_table.setSizePolicy(sizePolicy)
        self.per_table.setShowGrid(True)
        self.per_table.setCornerButtonEnabled(True)
        self.per_table.setObjectName("per_table")
        self.per_table.setColumnCount(3)
        self.per_table.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.per_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.per_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.per_table.setHorizontalHeaderItem(2, item)
        self.per_table.horizontalHeader().setDefaultSectionSize(102)
        self.per_table.horizontalHeader().setStretchLastSection(False)
        self.per_table.verticalHeader().setVisible(True)
        self.ap_power_widget = QtWidgets.QGraphicsView(self.sig_gen_tab)
        self.ap_power_widget.setGeometry(QtCore.QRect(1110, 9, 411, 411))
        self.ap_power_widget.setObjectName("ap_power_widget")
        self.ap_time_widget = QtWidgets.QGraphicsView(self.sig_gen_tab)
        self.ap_time_widget.setGeometry(QtCore.QRect(340, 10, 760, 411))
        self.ap_time_widget.setObjectName("ap_time_widget")
        self.line = QtWidgets.QFrame(self.sig_gen_tab)
        self.line.setGeometry(QtCore.QRect(127, 420, 1391, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.per_power_widget = QtWidgets.QGraphicsView(self.sig_gen_tab)
        self.per_power_widget.setGeometry(QtCore.QRect(1110, 440, 411, 411))
        self.per_power_widget.setObjectName("per_power_widget")
        self.per_time_widget = QtWidgets.QGraphicsView(self.sig_gen_tab)
        self.per_time_widget.setGeometry(QtCore.QRect(339, 440, 761, 411))
        self.per_time_widget.setObjectName("per_time_widget")
        self.checkBox = QtWidgets.QCheckBox(self.sig_gen_tab)
        self.checkBox.setGeometry(QtCore.QRect(23, 84, 131, 23))
        self.checkBox.setObjectName("checkBox")
        self.ap_gaussian_spinBox = QtWidgets.QDoubleSpinBox(self.sig_gen_tab)
        self.ap_gaussian_spinBox.setEnabled(False)
        self.ap_gaussian_spinBox.setGeometry(QtCore.QRect(160, 83, 81, 26))
        self.ap_gaussian_spinBox.setDecimals(3)
        self.ap_gaussian_spinBox.setMinimum(0.0)
        self.ap_gaussian_spinBox.setMaximum(1.0)
        self.ap_gaussian_spinBox.setSingleStep(0.001)
        self.ap_gaussian_spinBox.setObjectName("ap_gaussian_spinBox")
        self.label_6 = QtWidgets.QLabel(self.sig_gen_tab)
        self.label_6.setGeometry(QtCore.QRect(10, 700, 121, 17))
        self.label_6.setObjectName("label_6")
        self.line_2 = QtWidgets.QFrame(self.sig_gen_tab)
        self.line_2.setGeometry(QtCore.QRect(130, 700, 201, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_7 = QtWidgets.QLabel(self.sig_gen_tab)
        self.label_7.setGeometry(QtCore.QRect(20, 733, 121, 17))
        self.label_7.setObjectName("label_7")
        self.signal_duration_spinBox = QtWidgets.QDoubleSpinBox(self.sig_gen_tab)
        self.signal_duration_spinBox.setGeometry(QtCore.QRect(200, 730, 81, 26))
        self.signal_duration_spinBox.setDecimals(1)
        self.signal_duration_spinBox.setMinimum(1.0)
        self.signal_duration_spinBox.setMaximum(30.0)
        self.signal_duration_spinBox.setProperty("value", 10.0)
        self.signal_duration_spinBox.setObjectName("signal_duration_spinBox")
        self.epochduration_spinBox = QtWidgets.QDoubleSpinBox(self.sig_gen_tab)
        self.epochduration_spinBox.setGeometry(QtCore.QRect(200, 760, 81, 26))
        self.epochduration_spinBox.setDecimals(1)
        self.epochduration_spinBox.setMinimum(1.0)
        self.epochduration_spinBox.setMaximum(30.0)
        self.epochduration_spinBox.setObjectName("epochduration_spinBox")
        self.label_8 = QtWidgets.QLabel(self.sig_gen_tab)
        self.label_8.setGeometry(QtCore.QRect(20, 764, 131, 17))
        self.label_8.setObjectName("label_8")
        self.fs_spinBox = QtWidgets.QSpinBox(self.sig_gen_tab)
        self.fs_spinBox.setGeometry(QtCore.QRect(200, 819, 81, 30))
        self.fs_spinBox.setMinimum(500)
        self.fs_spinBox.setMaximum(5000)
        self.fs_spinBox.setSingleStep(100)
        self.fs_spinBox.setProperty("value", 1000)
        self.fs_spinBox.setObjectName("fs_spinBox")
        self.label_9 = QtWidgets.QLabel(self.sig_gen_tab)
        self.label_9.setGeometry(QtCore.QRect(20, 820, 171, 24))
        self.label_9.setObjectName("label_9")
        self.epochduration_spinBox_2 = QtWidgets.QDoubleSpinBox(self.sig_gen_tab)
        self.epochduration_spinBox_2.setGeometry(QtCore.QRect(200, 789, 81, 26))
        self.epochduration_spinBox_2.setDecimals(1)
        self.epochduration_spinBox_2.setMinimum(1.0)
        self.epochduration_spinBox_2.setMaximum(30.0)
        self.epochduration_spinBox_2.setObjectName("epochduration_spinBox_2")
        self.label_10 = QtWidgets.QLabel(self.sig_gen_tab)
        self.label_10.setGeometry(QtCore.QRect(20, 793, 131, 17))
        self.label_10.setObjectName("label_10")
        self.tabWidget.addTab(self.sig_gen_tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setAccessibleName("")
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        dialog.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(dialog)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1532, 22))
        self.menubar.setObjectName("menubar")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        dialog.setMenuBar(self.menubar)
        self.actionPSD_X_axis_log_scale = QtWidgets.QAction(dialog)
        self.actionPSD_X_axis_log_scale.setCheckable(True)
        self.actionPSD_X_axis_log_scale.setObjectName("actionPSD_X_axis_log_scale")
        self.actionPSD_Y_axis_log_scale = QtWidgets.QAction(dialog)
        self.actionPSD_Y_axis_log_scale.setCheckable(True)
        self.actionPSD_Y_axis_log_scale.setObjectName("actionPSD_Y_axis_log_scale")
        self.menuView.addAction(self.actionPSD_X_axis_log_scale)
        self.menuView.addAction(self.actionPSD_Y_axis_log_scale)
        self.menubar.addAction(self.menuView.menuAction())

        self.retranslateUi(dialog)
        self.tabWidget.setCurrentIndex(0)
        self.ap_stackedWidget.setCurrentIndex(0)
        self.checkBox.clicked['bool'].connect(self.ap_gaussian_spinBox.setEnabled) # type: ignore
        self.ap_model_comboBox.currentIndexChanged['int'].connect(self.ap_stackedWidget.setCurrentIndex) # type: ignore
        self.add_per_button.clicked.connect(dialog.add_per_signal) # type: ignore
        self.remove_per_button.clicked.connect(dialog.remove_per_signal) # type: ignore
        self.ap_fixed_intercept_spinBox.valueChanged['double'].connect(dialog.update_ap_view) # type: ignore
        self.ap_fixed_exponent_spinBox.valueChanged['double'].connect(dialog.update_ap_view) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "Unraveling of Neural Complexity Simulator"))
        self.label.setText(_translate("dialog", "Aperiodic signal"))
        self.remove_per_button.setText(_translate("dialog", "Remove"))
        self.ap_model_comboBox.setItemText(0, _translate("dialog", "Fixed 1/f"))
        self.ap_model_comboBox.setItemText(1, _translate("dialog", "Knee 1/f"))
        self.add_per_button.setText(_translate("dialog", "Add"))
        self.label_4.setText(_translate("dialog", "Exponent"))
        self.label_5.setText(_translate("dialog", "Intercept"))
        self.label_3.setText(_translate("dialog", "model"))
        self.label_2.setText(_translate("dialog", "Periodic Signal"))
        self.per_table.setSortingEnabled(True)
        item = self.per_table.horizontalHeaderItem(0)
        item.setText(_translate("dialog", "Freqeuncy"))
        item = self.per_table.horizontalHeaderItem(1)
        item.setText(_translate("dialog", "Amplitude"))
        item = self.per_table.horizontalHeaderItem(2)
        item.setText(_translate("dialog", "Peak width"))
        self.checkBox.setText(_translate("dialog", "add white noise"))
        self.ap_gaussian_spinBox.setToolTip(_translate("dialog", "Gaussian noise Standard deviation"))
        self.label_6.setText(_translate("dialog", "General Settings"))
        self.label_7.setText(_translate("dialog", "Signal duration (s)"))
        self.label_8.setText(_translate("dialog", "Epoch duration (s)"))
        self.label_9.setText(_translate("dialog", "Sampling frequency (Hz)"))
        self.label_10.setText(_translate("dialog", "Epoch overlap (s)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.sig_gen_tab), _translate("dialog", "Signal simulator"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("dialog", "Aperiodic estimator"))
        self.menuView.setTitle(_translate("dialog", "View"))
        self.actionPSD_X_axis_log_scale.setText(_translate("dialog", "PSD X-axis log scale"))
        self.actionPSD_Y_axis_log_scale.setText(_translate("dialog", "PSD Y-axis log scale"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    dialog = QtWidgets.QMainWindow()
    ui = Ui_dialog()
    ui.setupUi(dialog)
    dialog.show()
    sys.exit(app.exec_())