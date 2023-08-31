import sys
import app_framework as af
from PyQt5 import QtWidgets

if __name__ == '__main__':
    # Create GUI application
    app = QtWidgets.QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    form = af.MyApp(dpi)
    form.show()
    app.exec_()