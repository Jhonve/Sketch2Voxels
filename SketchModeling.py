import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, pyqtSlot

import os

class DrawWindow(QWidget):
    def __init__(self):
        super(DrawWindow, self).__init__()
        self.win_height = 512
        self.win_width = 512
        self.win_pos_x = 200
        self.win_pos_y = 200
        self.res_height = 256
        self.res_width = 256

        self.resize(self.win_height, self.win_width)
        self.move(self.win_pos_x, self.win_pos_y)
        self.setWindowTitle("Sketch Modeling")
        self.setMouseTracking(False)
        
        # to save positions
        self.pos_xy = []
        self.line_num = 0
        self.line_index = []
        self.line_index.append(0)
        self.initUI()

    def initUI(self):
        self.gen_button_pos_x = 286
        self.gen_button_pos_y = 482
        self.gen_button = QPushButton("Generate", self)
        self.gen_button.setToolTip("Generate 3D voxels")
        self.gen_button.move(self.gen_button_pos_x, self.gen_button_pos_y)
        self.gen_button.clicked.connect(self.onGenClick)

        self.del_button_pos_x = 176
        self.del_button_pos_y = 482
        self.del_button = QPushButton("Recall", self)
        self.del_button.setToolTip("Recall one step")
        self.del_button.move(self.del_button_pos_x, self.del_button_pos_y)
        self.del_button.clicked.connect(self.onDelClick)

        self.remove_button_pos_x = 422
        self.remove_button_pos_y = 10
        self.remove_button = QPushButton("Remove", self)
        self.remove_button.setToolTip("Remove all")
        self.remove_button.move(self.remove_button_pos_x, self.remove_button_pos_y)
        self.remove_button.clicked.connect(self.onRemClick)

    @pyqtSlot()
    def onGenClick(self):
        if(self.line_num > 0):
            image = np.ones([self.win_height, self.win_width])
            image = np.uint8(image * 255)
        
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                image = cv2.line(image, (point_start[0], point_start[1]), 
                        (point_end[0], point_end[1]), 0, 1, 8)
                point_start = point_end

            image = cv2.resize(image, (self.res_height, self.res_width), cv2.INTER_LINEAR)
            cv2.imwrite("./TestData/1/sketch.jpg", image)
            while(True):
                try:    
                    with open("./TestData/state.txt", "w") as state_file:
                        state_file.write("1\n0\n0\n0")
                        state_file.close()
                        break
                except PermissionError as e:
                    print("PermissionError")

            print("Generating now")

            while(True):
                with open("./TestData/state.txt", "r") as state_file:
                    line_list = state_file.readlines()
                    if(len(line_list) == 0):
                        continue
                    if(line_list[2] == "1\n"):
                        state_file.close()
                        os.system("python3 Visualization/visualize.py TestData/1/voxels.mat -cm")
                        break
                    else:
                        state_file.close()
            print("Generate Done!")
        else:
            print("Draw first.")

    @pyqtSlot()
    def onDelClick(self):
        if(self.line_num > 0):
            self.pos_xy = self.pos_xy[:self.line_index[self.line_num - 1]]
            self.line_index.pop(self.line_num)
            self.line_num = self.line_num - 1
        else:
            print("Draw first.")

        self.update()

    @pyqtSlot()
    def onRemClick(self):
        if(self.line_num > 0):
            self.pos_xy = []
            self.line_num = 0
            self.line_index = []
            self.line_index.append(0)
        else:
            print("Draw first.")
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        pos_tmp = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        pos_tmp = (-1, -1)
        self.pos_xy.append(pos_tmp)
        self.line_num = self.line_num + 1
        self.line_index.append(len(self.pos_xy))

        self.update()

if __name__ == "__main__":
    sketch_model_app = QApplication(sys.argv)
    draw_window = DrawWindow()
    draw_window.show()
    sketch_model_app.exec_()
