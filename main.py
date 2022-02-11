import sys
import sqlite3
import design
from PyQt5 import QtWidgets, QtCore, QtGui
import bar_detector
import cv2
import time
from imutils.video import FileVideoStream, FPS
from datetime import datetime
import webbrowser

SAVE_BARCODES = True

conn = sqlite3.connect('history.db')
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS barcode(
   id INTEGER PRIMARY KEY AUTOINCREMENT,
   code TEXT,
   type TEXT,
   date TEXT);
""")
conn.commit()
conn.close()


class Thread(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QtGui.QPixmap)

    def run(self):
        fvs = FileVideoStream('staff/main_test3.mp4').start()
        fps = FPS().start()
        time.sleep(1.0)
        while True:
            frame = fvs.read()
            if frame is None:
                break
            answer, image = bar_detector.start(frame)
            rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
            p = convertToQtFormat.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
            self.changePixmap.emit(p)
            if answer != "cannot encode" and SAVE_BARCODES:
                date = datetime.today().strftime("%Y/%m/%d %H:%M:%S")
                conn = sqlite3.connect('history.db')
                cur = conn.cursor()
                cur.execute("""SELECT * FROM barcode WHERE code=?""", (answer[0],))
                rows = cur.fetchall()
                if rows:
                    cur.execute("""UPDATE barcode set date=? WHERE code=?""", (date, answer[0]))
                    conn.commit()
                else:
                    cur.execute("""INSERT INTO barcode (code, type, date)
                    VALUES (?, ?, ?)""", (answer[0], answer[1], date))
                    conn.commit()
                conn.close()
            if answer != "cannot encode":
                break
            fps.update()
        fps.stop()
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        fvs.stop()


class ExampleApp(QtWidgets.QMainWindow, design.Ui_BarcodeScanner):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.centralwidget.installEventFilter(self)
        self.HistoryList.setHeaderLabels(('Код', 'Тип', 'Дата и время'))
        self.SaveButton.setChecked(True)
        self.tabWidget.setCurrentIndex(0)
        self.stream = Thread()
        self.update_history()
        self.stream.finished.connect(self.update_history)
        self.ScanButton.clicked.connect(self.start_scanner)
        self.SaveButton.clicked.connect(self.save_in_history)
        self.ClearButton.clicked.connect(self.clear_database)
        self.SearchButton.clicked.connect(self.search_in_brouser)
        self.Screen.setScaledContents(True)
        self.Screen.hasHeightForWidth()

    def start_scanner(self):
        self.stream.changePixmap.connect(self.Screen.setPixmap)
        self.stream.start()

    def save_in_history(self):
        global SAVE_BARCODES
        if self.SaveButton.isChecked():
            SAVE_BARCODES = True
        else:
            SAVE_BARCODES = False

    def update_history(self):
        conn = sqlite3.connect('history.db')
        cur = conn.cursor()
        cur.execute("""SELECT * FROM barcode ORDER BY date DESC;""")
        records = cur.fetchall()
        self.HistoryList.clear()
        items = []
        for record in records:
            item = QtWidgets.QTreeWidgetItem([record[1], record[2], record[3]])
            items.append(item)
        self.HistoryList.insertTopLevelItems(0, items)
        conn.close()

    def search_in_brouser(self):
        if self.HistoryList.selectedItems():
            code = self.HistoryList.selectedItems()[0].text(0)
            url = f'https://www.google.com/m/products?q={code}'
            webbrowser.open_new_tab(url)

    def clear_database(self):
        conn = sqlite3.connect('history.db')
        cur = conn.cursor()
        cur.execute("""DELETE FROM barcode;""")
        conn.commit()
        conn.close()
        self.HistoryList.clear()

    def eventFilter(self, watched, event):
        ratio = 640/480
        if watched == self.centralwidget and event.type() == QtCore.QEvent.Resize and \
                self.centralwidget.width() > 0 and self.centralwidget.height() > 0:
            central_ratio = self.centralwidget.width() / self.centralwidget.height()
            if central_ratio != ratio:
                new_height = int(self.centralwidget.width() / ratio)
                self.centralwidget.resize(self.centralwidget.width(), new_height)
        return super(ExampleApp, self).eventFilter(watched, event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()

