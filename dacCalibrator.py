import json
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports

from PySide6 import QtCore, QtGui, QtWidgets


class SerialWorker(QtCore.QObject):
  pressure_received = QtCore.Signal(float)
  status_changed = QtCore.Signal(str)
  connected_changed = QtCore.Signal(bool)

  def __init__(self):
    super().__init__()
    self.serial_port = None
    self._running = False
    self._thread = None

  def connect_port(self, port_name, baud_rate=115200):
    try:
      self.serial_port = serial.Serial(port_name, baud_rate, timeout=0.1)
      self._running = True
      self._thread = threading.Thread(target=self._read_loop, daemon=True)
      self._thread.start()
      self.status_changed.emit(f"Connected to {port_name} @ {baud_rate}")
      self.connected_changed.emit(True)
    except Exception as exc:
      self.status_changed.emit(f"Connect failed: {exc}")
      self.connected_changed.emit(False)

  def disconnect_port(self):
    self._running = False

    if self._thread and self._thread.is_alive():
      self._thread.join(timeout=1.0)

    if self.serial_port:
      try:
        if self.serial_port.is_open:
          self.serial_port.close()
      except Exception:
        pass

    self.serial_port = None
    self.status_changed.emit("Disconnected")
    self.connected_changed.emit(False)

  def send_line(self, line):
    if not self.serial_port or not self.serial_port.is_open:
      self.status_changed.emit("Not connected")
      return

    try:
      payload = (line.strip() + "\n").encode("utf-8")
      self.serial_port.write(payload)
      self.status_changed.emit(f"Sent: {line}")
    except Exception as exc:
      self.status_changed.emit(f"Send failed: {exc}")

  def _read_loop(self):
    while self._running and self.serial_port and self.serial_port.is_open:
      try:
        raw = self.serial_port.readline()
        if not raw:
          time.sleep(0.01)
          continue

        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
          continue

        if line.startswith("PRESSURE:"):
          value_text = line.split(":", 1)[1].strip()
          pressure = float(value_text)
          self.pressure_received.emit(pressure)
        else:
          self.status_changed.emit(f"RX: {line}")

      except Exception as exc:
        self.status_changed.emit(f"Read error: {exc}")
        break

    self.connected_changed.emit(False)


class DraggableCalibrationGraph(pg.GraphItem):
  pointMoved = QtCore.Signal(int, float, float)

  def __init__(self):
    super().__init__()
    self.data = {}
    self.drag_index = None
    self.scatter.sigClicked.connect(self._on_click)

  def set_points(self, points):
    self.points = [list(point) for point in points]
    pos = []
    for x_value, y_value in self.points:
      pos.append((float(x_value), float(y_value)))

    self.data = {
      "pos": np.array(pos, dtype=float),
      "size": 14,
      "symbol": "o",
      "pxMode": True,
      "pen": pg.mkPen(width=1),
      "brush": pg.mkBrush(50, 150, 255, 180)
    }
    self.setData(**self.data)

  def _on_click(self, plot, points):
    if points:
      self.drag_index = points[0].index()

  def mouseDragEvent(self, ev):
    if self.drag_index is None:
      ev.ignore()
      return

    if ev.button() != QtCore.Qt.MouseButton.LeftButton:
      ev.ignore()
      return

    if ev.isStart():
      ev.accept()
      return

    if ev.isFinish():
      self.drag_index = None
      ev.accept()
      return

    ev.accept()

    mouse_point = self.getViewBox().mapSceneToView(ev.scenePos())
    x_value = float(mouse_point.x())
    y_value = float(mouse_point.y())

    # Clamp Y to 0-5V
    y_value = max(0.0, min(5.0, y_value))

    # Clamp X to keep points ordered
    left_limit = -float("inf")
    right_limit = float("inf")

    if self.drag_index > 0:
      left_limit = self.points[self.drag_index - 1][0]
    if self.drag_index < len(self.points) - 1:
      right_limit = self.points[self.drag_index + 1][0]

    x_value = max(left_limit, min(right_limit, x_value))

    self.points[self.drag_index][0] = x_value
    self.points[self.drag_index][1] = y_value
    self.set_points(self.points)
    self.pointMoved.emit(self.drag_index, x_value, y_value)


class MainWindow(QtWidgets.QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("Handbrake Calibration Tool")
    self.resize(1100, 700)

    self.current_pressure = 0.0
    self.min_pressure = 0.0
    self.max_pressure = 100.0

    self.curve_points = [
      [0.0, 0.0],
      [25.0, 1.25],
      [50.0, 2.5],
      [75.0, 3.75],
      [100.0, 5.0]
    ]

    self.serial_worker = SerialWorker()
    self.serial_worker.pressure_received.connect(self.on_pressure_received)
    self.serial_worker.status_changed.connect(self.set_status)
    self.serial_worker.connected_changed.connect(self.on_connection_changed)

    self._building_table = False

    self._build_ui()
    self.refresh_ports()
    self.refresh_graph()

  def _build_ui(self):
    central = QtWidgets.QWidget()
    self.setCentralWidget(central)

    main_layout = QtWidgets.QVBoxLayout(central)

    top_row = QtWidgets.QHBoxLayout()
    main_layout.addLayout(top_row)

    self.port_combo = QtWidgets.QComboBox()
    self.refresh_ports_button = QtWidgets.QPushButton("Refresh Ports")
    self.connect_button = QtWidgets.QPushButton("Connect")
    self.disconnect_button = QtWidgets.QPushButton("Disconnect")
    self.disconnect_button.setEnabled(False)

    top_row.addWidget(QtWidgets.QLabel("COM Port:"))
    top_row.addWidget(self.port_combo, 1)
    top_row.addWidget(self.refresh_ports_button)
    top_row.addWidget(self.connect_button)
    top_row.addWidget(self.disconnect_button)

    pressure_group = QtWidgets.QGroupBox("Live Pressure")
    pressure_layout = QtWidgets.QGridLayout(pressure_group)
    main_layout.addWidget(pressure_group)

    self.pressure_value_edit = QtWidgets.QLineEdit("0.00")
    self.pressure_value_edit.setReadOnly(True)

    self.min_value_edit = QtWidgets.QLineEdit(f"{self.min_pressure:.2f}")

    self.max_value_edit = QtWidgets.QLineEdit(f"{self.max_pressure:.2f}")

    self.apply_range_button = QtWidgets.QPushButton("Apply Range")

    pressure_layout.addWidget(QtWidgets.QLabel("Current Pressure:"), 0, 0)
    pressure_layout.addWidget(self.pressure_value_edit, 0, 1)
    pressure_layout.addWidget(QtWidgets.QLabel("Min Pressure (manual):"), 1, 0)
    pressure_layout.addWidget(self.min_value_edit, 1, 1)
    pressure_layout.addWidget(QtWidgets.QLabel("Max Pressure (manual):"), 2, 0)
    pressure_layout.addWidget(self.max_value_edit, 2, 1)
    pressure_layout.addWidget(self.apply_range_button, 1, 2, 2, 1)

    splitter = QtWidgets.QSplitter()
    splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
    main_layout.addWidget(splitter, 1)

    graph_container = QtWidgets.QWidget()
    graph_layout = QtWidgets.QVBoxLayout(graph_container)
    splitter.addWidget(graph_container)

    self.plot_widget = pg.PlotWidget()
    self.plot_widget.setLabel("bottom", "Pressure")
    self.plot_widget.setLabel("left", "Volts")
    self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
    self.plot_widget.setYRange(0, 5, padding=0.05)
    self.plot_widget.setXRange(self.min_pressure, self.max_pressure, padding=0.05)
    self.plot_widget.setMouseEnabled(x=False, y=False)

    graph_layout.addWidget(self.plot_widget)

    self.curve_line = self.plot_widget.plot([], [], pen=pg.mkPen(width=2))
    self.graph_points = DraggableCalibrationGraph()
    self.graph_points.pointMoved.connect(self.on_graph_point_moved)
    self.plot_widget.addItem(self.graph_points)

    table_container = QtWidgets.QWidget()
    table_layout = QtWidgets.QVBoxLayout(table_container)
    splitter.addWidget(table_container)

    self.points_table = QtWidgets.QTableWidget()
    self.points_table.setColumnCount(2)
    self.points_table.setHorizontalHeaderLabels(["Pressure", "Voltage"])
    self.points_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
    self.points_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    self.points_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    table_layout.addWidget(self.points_table)

    edit_row = QtWidgets.QHBoxLayout()
    table_layout.addLayout(edit_row)
    self.add_point_button = QtWidgets.QPushButton("Add Point")
    self.remove_point_button = QtWidgets.QPushButton("Remove Selected")
    edit_row.addWidget(self.add_point_button)
    edit_row.addWidget(self.remove_point_button)

    button_row = QtWidgets.QHBoxLayout()
    table_layout.addLayout(button_row)

    self.send_curve_button = QtWidgets.QPushButton("Send Curve")
    self.save_eeprom_button = QtWidgets.QPushButton("Save to EEPROM")
    button_row.addWidget(self.send_curve_button)
    button_row.addWidget(self.save_eeprom_button)

    self.status_label = QtWidgets.QLabel("Ready")
    main_layout.addWidget(self.status_label)

    self.refresh_ports_button.clicked.connect(self.refresh_ports)
    self.connect_button.clicked.connect(self.connect_serial)
    self.disconnect_button.clicked.connect(self.disconnect_serial)
    self.apply_range_button.clicked.connect(self.apply_pressure_range)
    self.add_point_button.clicked.connect(self.add_point)
    self.remove_point_button.clicked.connect(self.remove_selected_point)
    self.send_curve_button.clicked.connect(self.send_curve_to_device)
    self.save_eeprom_button.clicked.connect(self.save_to_eeprom)
    self.points_table.itemChanged.connect(self.on_table_item_changed)

    self.populate_table()

  def refresh_ports(self):
    self.port_combo.clear()
    ports = serial.tools.list_ports.comports()
    for port in ports:
      self.port_combo.addItem(port.device)

    if self.port_combo.count() == 0:
      self.port_combo.addItem("No ports found")

  def connect_serial(self):
    port_name = self.port_combo.currentText()
    if not port_name or port_name == "No ports found":
      self.set_status("No serial port selected")
      return

    self.serial_worker.connect_port(port_name)

  def disconnect_serial(self):
    self.serial_worker.disconnect_port()

  def on_connection_changed(self, connected):
    self.connect_button.setEnabled(not connected)
    self.disconnect_button.setEnabled(connected)

  def on_pressure_received(self, pressure):
    self.current_pressure = pressure
    self.pressure_value_edit.setText(f"{pressure:.2f}")

  def apply_pressure_range(self):
    try:
      min_value = float(self.min_value_edit.text().strip())
      max_value = float(self.max_value_edit.text().strip())
    except ValueError:
      self.set_status("Invalid min/max pressure values")
      self.min_value_edit.setText(f"{self.min_pressure:.2f}")
      self.max_value_edit.setText(f"{self.max_pressure:.2f}")
      return

    if min_value >= max_value:
      self.set_status("Min pressure must be less than max pressure")
      self.min_value_edit.setText(f"{self.min_pressure:.2f}")
      self.max_value_edit.setText(f"{self.max_pressure:.2f}")
      return

    self.min_pressure = round(min_value, 2)
    self.max_pressure = round(max_value, 2)
    self.min_value_edit.setText(f"{self.min_pressure:.2f}")
    self.max_value_edit.setText(f"{self.max_pressure:.2f}")
    self.refresh_graph()
    self.set_status("Pressure range updated")

  def normalize_pressure_order(self):
    self.curve_points.sort(key=lambda point: point[0])

  def expand_pressure_range_to_points(self):
    if not self.curve_points:
      return

    x_values = [point[0] for point in self.curve_points]
    next_min = self.min_pressure
    next_max = self.max_pressure

    if min(x_values) < next_min:
      next_min = round(min(x_values), 2)
    if max(x_values) > next_max:
      next_max = round(max(x_values), 2)

    if next_min != self.min_pressure or next_max != self.max_pressure:
      self.min_pressure = next_min
      self.max_pressure = next_max
      self.min_value_edit.setText(f"{self.min_pressure:.2f}")
      self.max_value_edit.setText(f"{self.max_pressure:.2f}")

  def populate_table(self):
    self._building_table = True
    self.points_table.blockSignals(True)
    self.points_table.setRowCount(len(self.curve_points))

    for row_index, (pressure, voltage) in enumerate(self.curve_points):
      pressure_item = QtWidgets.QTableWidgetItem(f"{pressure:.2f}")
      voltage_item = QtWidgets.QTableWidgetItem(f"{voltage:.2f}")

      pressure_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
      voltage_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

      self.points_table.setItem(row_index, 0, pressure_item)
      self.points_table.setItem(row_index, 1, voltage_item)

    self.points_table.blockSignals(False)
    self._building_table = False

  def add_point(self):
    row = self.points_table.currentRow()
    point_count = len(self.curve_points)

    if point_count < 2:
      self.curve_points.append([0.0, 0.0])
      self.populate_table()
      self.refresh_graph()
      return

    if 0 <= row < point_count - 1:
      left = self.curve_points[row]
      right = self.curve_points[row + 1]
      insert_index = row + 1
    elif row == point_count - 1:
      left = self.curve_points[row - 1]
      right = self.curve_points[row]
      insert_index = row
    else:
      # No selected row: insert in the middle of the largest pressure gap.
      largest_gap = -1.0
      insert_index = 1
      left = self.curve_points[0]
      right = self.curve_points[1]
      for idx in range(point_count - 1):
        gap = self.curve_points[idx + 1][0] - self.curve_points[idx][0]
        if gap > largest_gap:
          largest_gap = gap
          left = self.curve_points[idx]
          right = self.curve_points[idx + 1]
          insert_index = idx + 1

    new_x = round((left[0] + right[0]) / 2.0, 2)
    new_y = round((left[1] + right[1]) / 2.0, 2)
    self.curve_points.insert(insert_index, [new_x, new_y])
    self.expand_pressure_range_to_points()

    self.populate_table()
    self.points_table.selectRow(insert_index)
    self.refresh_graph()

  def remove_selected_point(self):
    if len(self.curve_points) <= 2:
      self.set_status("At least two points are required")
      return

    row = self.points_table.currentRow()
    if row < 0 or row >= len(self.curve_points):
      self.set_status("Select a row to remove")
      return

    del self.curve_points[row]
    self.populate_table()

    if self.curve_points:
      next_row = min(row, len(self.curve_points) - 1)
      self.points_table.selectRow(next_row)

    self.refresh_graph()

  def on_table_item_changed(self, item):
    if self._building_table:
      return

    row = item.row()
    column = item.column()

    try:
      value = float(item.text())
    except ValueError:
      self.populate_table()
      return

    if column == 0:
      self.curve_points[row][0] = value
    elif column == 1:
      self.curve_points[row][1] = max(0.0, min(5.0, value))

    self.normalize_pressure_order()
    self.expand_pressure_range_to_points()
    self.populate_table()
    self.refresh_graph()

  def refresh_graph(self):
    x_values = [point[0] for point in self.curve_points]
    y_values = [point[1] for point in self.curve_points]
    x_padding = max((self.max_pressure - self.min_pressure) * 0.05, 1.0)
    self.plot_widget.setXRange(self.min_pressure - x_padding, self.max_pressure + x_padding, padding=0.0)
    self.plot_widget.setYRange(0.0, 5.0, padding=0.05)

    self.curve_line.setData(x_values, y_values)
    self.graph_points.set_points(self.curve_points)

  def on_graph_point_moved(self, index, x_value, y_value):
    self.curve_points[index][0] = round(x_value, 2)
    self.curve_points[index][1] = round(y_value, 2)
    self.expand_pressure_range_to_points()
    self.populate_table()
    self.refresh_graph()

  def send_curve_to_device(self):
    payload = {
      "points": self.curve_points
    }
    line = "SET_CURVE:" + json.dumps(payload["points"])
    self.serial_worker.send_line(line)

  def save_to_eeprom(self):
    self.serial_worker.send_line("SAVE")

  def set_status(self, message):
    self.status_label.setText(message)

  def closeEvent(self, event):
    self.serial_worker.disconnect_port()
    super().closeEvent(event)


def main():
  app = QtWidgets.QApplication(sys.argv)
  app.setApplicationName("Handbrake Calibration Tool")

  pg.setConfigOptions(antialias=True)

  window = MainWindow()
  window.show()

  sys.exit(app.exec())


if __name__ == "__main__":
  main()
