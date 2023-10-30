from label_drawing_to_array.label_drawing_to_array import LabelDrawingToArray
from modal_dialog.gui_modal import Ui_dialog_convert_graphics_to_values

from modal_dialog.graph import Graph
from modal_dialog.drawer import Drawer

from PyQt5.QtWidgets import (QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt


# Класс модального окна
class DateDialog(QDialog, Ui_dialog_convert_graphics_to_values):
    def __init__(self, parent=None):
        # Создание окна и загрузка интерфейса
        super(DateDialog, self).__init__(parent)
        self.setupUi(self)

        # Область рисования
        self.label_field_antennas = LabelDrawingToArray()
        self.layout_plot_1.addWidget(self.label_field_antennas)

        # График для отображения / проверки перевода
        self.graph = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2
        )

        # Кнопки Ок и Отмена
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout_message_buttons.addWidget(buttons)

        # Сигналы
        self.pushButton_preview.clicked.connect(self.preview)
        self.pushButton_clear.clicked.connect(self.label_field_antennas.cleaning_and_mesh)

    # Статичный метод для создания диалога и возврата результата
    @staticmethod
    def get_array(parent=None):
        dialog = DateDialog(parent)
        result = dialog.exec_()
        coordinate_array = dialog.label_field_antennas.graph_in_distribution()
        return coordinate_array, result == QDialog.Accepted

    # Передпросмотр. Переводит рисунок в распределение и выводит в виде графика для проверки
    def preview(self):
        # Запрос распределения
        coordinate_array = self.label_field_antennas.graph_in_distribution()
        # Проверка на пустоту
        if coordinate_array is None:
            return
        # Отрисовка
        Drawer.graph(self.graph, coordinate_array)
