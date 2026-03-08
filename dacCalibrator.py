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
  dragStateChanged = QtCore.Signal(int, bool)
  voltageLimitHit = QtCore.Signal()
  pressureLimitHit = QtCore.Signal()

  def __init__(self):
    super().__init__()
    self.data = {}
    self.drag_index = None
    self.drag_offset = None
    self.hover_index = None
    self.max_voltage = 5.0
    self._drag_hit_voltage_limit = False
    self._drag_hit_pressure_limit = False
    self.setAcceptHoverEvents(True)

  def set_max_voltage(self, max_voltage):
    self.max_voltage = max(0.0, min(5.0, float(max_voltage)))

  def _build_point_brushes(self, count):
    brushes = []
    for index in range(count):
      if self.drag_index == index:
        brushes.append(pg.mkBrush(70, 190, 90, 220))
      elif self.hover_index == index:
        brushes.append(pg.mkBrush(255, 190, 40, 220))
      else:
        brushes.append(pg.mkBrush(50, 150, 255, 180))
    return brushes

  def _extract_point_index(self, point):
    point_data = point.data()
    if isinstance(point_data, np.ndarray):
      if point_data.size == 0:
        return None
      return int(point_data.flat[0])
    return int(point_data)

  def _apply_point_brushes(self):
    if "pos" not in self.data:
      return
    self.data["brush"] = self._build_point_brushes(len(self.data["pos"]))
    self.setData(**self.data)

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
      "data": list(range(len(pos))),
      "pen": pg.mkPen(width=1),
      "brush": self._build_point_brushes(len(pos))
    }
    self.setData(**self.data)

  def hoverEvent(self, ev):
    if self.drag_index is not None:
      ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
      return

    if ev.isExit():
      next_hover = None
    else:
      hovered = self.scatter.pointsAt(ev.pos())
      next_hover = self._extract_point_index(hovered[0]) if hovered.size > 0 else None

    if next_hover != self.hover_index:
      self.hover_index = next_hover
      self._apply_point_brushes()

    ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)

  def mouseDragEvent(self, ev):
    if ev.button() != QtCore.Qt.MouseButton.LeftButton:
      ev.ignore()
      return

    if ev.isStart():
      clicked_points = self.scatter.pointsAt(ev.buttonDownPos())
      if not clicked_points:
        ev.ignore()
        return

      self.drag_index = self._extract_point_index(clicked_points[0])
      if self.drag_index is None:
        ev.ignore()
        return

      drag_index = self.drag_index
      self.drag_offset = self.data["pos"][drag_index] - np.array(
        [ev.buttonDownPos().x(), ev.buttonDownPos().y()],
        dtype=float
      )
      self._drag_hit_voltage_limit = False
      self._drag_hit_pressure_limit = False
      self._apply_point_brushes()
      self.dragStateChanged.emit(drag_index, True)
      ev.accept()
      return

    if ev.isFinish():
      last_drag_index = self.drag_index if self.drag_index is not None else -1
      if self._drag_hit_voltage_limit:
        self.voltageLimitHit.emit()
      if self._drag_hit_pressure_limit:
        self.pressureLimitHit.emit()
      self.drag_index = None
      self.drag_offset = None
      self._drag_hit_voltage_limit = False
      self._drag_hit_pressure_limit = False
      self._apply_point_brushes()
      self.dragStateChanged.emit(last_drag_index, False)
      ev.accept()
      return

    if self.drag_index is None:
      ev.ignore()
      return

    ev.accept()

    drag_index = self.drag_index
    mouse_pos = np.array([ev.pos().x(), ev.pos().y()], dtype=float)
    next_pos = mouse_pos + self.drag_offset
    x_value = float(next_pos[0])
    y_value = float(next_pos[1])

    if self.max_voltage < 5.0 and y_value > self.max_voltage:
      self._drag_hit_voltage_limit = True
    # Clamp Y to 0-effective max volts
    y_value = max(0.0, min(self.max_voltage, y_value))

    # Clamp X to keep points ordered
    left_limit = -float("inf")
    right_limit = float("inf")

    if drag_index > 0:
      left_limit = self.points[drag_index - 1][0]
    if drag_index < len(self.points) - 1:
      right_limit = self.points[drag_index + 1][0]

    if x_value < left_limit or x_value > right_limit:
      self._drag_hit_pressure_limit = True
    x_value = max(left_limit, min(right_limit, x_value))

    self.points[drag_index][0] = x_value
    self.points[drag_index][1] = y_value
    self.set_points(self.points)
    self.pointMoved.emit(drag_index, x_value, y_value)


class MainWindow(QtWidgets.QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("Handbrake Calibration Tool")
    self.resize(1100, 700)

    self.current_pressure = 0.0
    self.min_pressure = 0.0
    self.max_pressure = 100.0
    self.scale = 1.0

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
    self._updating_range_fields = False
    self._restoring_state = False
    self._drag_start_state = None
    self._history = []
    self._history_index = -1
    self._history_limit = 25

    self._build_ui()
    self.refresh_ports()
    self.refresh_graph()
    self._record_history_state()

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

    self.scale_value_edit = QtWidgets.QLineEdit(f"{self.scale:.2f}")

    pressure_layout.addWidget(QtWidgets.QLabel("Current Pressure:"), 0, 0)
    pressure_layout.addWidget(self.pressure_value_edit, 0, 1)
    pressure_layout.addWidget(QtWidgets.QLabel("Min Pressure (manual):"), 1, 0)
    pressure_layout.addWidget(self.min_value_edit, 1, 1)
    pressure_layout.addWidget(QtWidgets.QLabel("Max Pressure (manual):"), 2, 0)
    pressure_layout.addWidget(self.max_value_edit, 2, 1)
    pressure_layout.addWidget(QtWidgets.QLabel("Scale:"), 3, 0)
    pressure_layout.addWidget(self.scale_value_edit, 3, 1)

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
    self.graph_points.dragStateChanged.connect(self.on_graph_drag_state_changed)
    self.graph_points.voltageLimitHit.connect(self.show_voltage_limit_popup)
    self.graph_points.pressureLimitHit.connect(self.show_pressure_limit_popup)
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

    self.undo_button = QtWidgets.QPushButton("Undo")
    self.redo_button = QtWidgets.QPushButton("Redo")
    self.save_configuration_button = QtWidgets.QPushButton("Save Configuration")
    button_row.addWidget(self.undo_button)
    button_row.addWidget(self.redo_button)
    button_row.addWidget(self.save_configuration_button)

    self.status_label = QtWidgets.QLabel("Ready")
    main_layout.addWidget(self.status_label)

    self.refresh_ports_button.clicked.connect(self.refresh_ports)
    self.connect_button.clicked.connect(self.connect_serial)
    self.disconnect_button.clicked.connect(self.disconnect_serial)
    self.min_value_edit.textChanged.connect(self.on_range_or_scale_changed)
    self.max_value_edit.textChanged.connect(self.on_range_or_scale_changed)
    self.scale_value_edit.textChanged.connect(self.on_range_or_scale_changed)
    self.min_value_edit.editingFinished.connect(self.on_range_or_scale_edit_finished)
    self.max_value_edit.editingFinished.connect(self.on_range_or_scale_edit_finished)
    self.scale_value_edit.editingFinished.connect(self.on_range_or_scale_edit_finished)
    self.add_point_button.clicked.connect(self.add_point)
    self.remove_point_button.clicked.connect(self.remove_selected_point)
    self.undo_button.clicked.connect(self.undo_change)
    self.redo_button.clicked.connect(self.redo_change)
    self.save_configuration_button.clicked.connect(self.save_configuration)
    self.points_table.itemChanged.connect(self.on_table_item_changed)

    self.populate_table()
    self._update_history_buttons()

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

  def on_range_or_scale_changed(self):
    if self._updating_range_fields or self._restoring_state:
      return

    previous_scale = self.scale

    try:
      min_value = float(self.min_value_edit.text().strip())
      max_value = float(self.max_value_edit.text().strip())
      scale_value = float(self.scale_value_edit.text().strip())
    except ValueError:
      return

    if scale_value <= 0:
      return

    next_scale = round(scale_value, 4)
    scale_changed = abs(next_scale - previous_scale) > 1e-9

    if scale_changed:
      scale_factor = previous_scale / next_scale
      for index, point in enumerate(self.curve_points):
        self.curve_points[index][0] = round(point[0] * scale_factor, 2)
      self.normalize_pressure_order()

    self.min_pressure = round(min_value, 2)
    self.max_pressure = round(max_value, 2)
    self.scale = next_scale
    self.enforce_voltage_limit()
    self.populate_table()
    self.refresh_graph()

  def on_range_or_scale_edit_finished(self):
    self._record_history_state()

  def _capture_state(self):
    return {
      "curve_points": [[float(x), float(y)] for x, y in self.curve_points],
      "min_pressure": float(self.min_pressure),
      "max_pressure": float(self.max_pressure),
      "scale": float(self.scale)
    }

  def _states_equal(self, left_state, right_state):
    return left_state == right_state

  def _update_history_buttons(self):
    if not hasattr(self, "undo_button") or not hasattr(self, "redo_button"):
      return
    self.undo_button.setEnabled(self._history_index > 0)
    self.redo_button.setEnabled(self._history_index < len(self._history) - 1)

  def _record_history_state(self):
    if self._restoring_state:
      return

    state = self._capture_state()

    if self._history_index >= 0 and self._states_equal(state, self._history[self._history_index]):
      self._update_history_buttons()
      return

    if self._history_index < len(self._history) - 1:
      self._history = self._history[:self._history_index + 1]

    self._history.append(state)
    if len(self._history) > self._history_limit:
      overflow = len(self._history) - self._history_limit
      self._history = self._history[overflow:]

    self._history_index = len(self._history) - 1
    self._update_history_buttons()

  def _apply_state(self, state):
    self._restoring_state = True
    self.curve_points = [[float(x), float(y)] for x, y in state["curve_points"]]
    self.min_pressure = float(state["min_pressure"])
    self.max_pressure = float(state["max_pressure"])
    self.scale = float(state["scale"])

    self._updating_range_fields = True
    self.min_value_edit.setText(f"{self.min_pressure:.2f}")
    self.max_value_edit.setText(f"{self.max_pressure:.2f}")
    self.scale_value_edit.setText(f"{self.scale:.2f}")
    self._updating_range_fields = False

    self.populate_table()
    self.refresh_graph()
    self._restoring_state = False
    self._update_history_buttons()

  def undo_change(self):
    if self._history_index <= 0:
      return
    self._history_index -= 1
    self._apply_state(self._history[self._history_index])

  def redo_change(self):
    if self._history_index >= len(self._history) - 1:
      return
    self._history_index += 1
    self._apply_state(self._history[self._history_index])

  def normalize_pressure_order(self):
    self.curve_points.sort(key=lambda point: point[0])

  def effective_max_voltage(self):
    return max(0.0, min(5.0, 5.0 * self.scale))

  def enforce_voltage_limit(self):
    max_voltage = self.effective_max_voltage()
    for index, point in enumerate(self.curve_points):
      self.curve_points[index][1] = round(max(0.0, min(max_voltage, point[1])), 2)

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
      self._updating_range_fields = True
      self.min_value_edit.setText(f"{self.min_pressure:.2f}")
      self.max_value_edit.setText(f"{self.max_pressure:.2f}")
      self._updating_range_fields = False

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
    self._updating_range_fields = False

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
    self._record_history_state()

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
    self._record_history_state()

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
      max_voltage = self.effective_max_voltage()
      if value > max_voltage:
        self.curve_points[row][1] = round(max_voltage, 2)
        if max_voltage < 5.0:
          self.show_voltage_limit_popup()
      else:
        self.curve_points[row][1] = max(0.0, value)

    self.normalize_pressure_order()
    self.expand_pressure_range_to_points()
    self.populate_table()
    self.refresh_graph()
    self._record_history_state()

  def refresh_graph(self):
    x_values = [point[0] for point in self.curve_points]
    y_values = [point[1] for point in self.curve_points]
    view_min = min(self.min_pressure, self.max_pressure)
    view_max = max(self.min_pressure, self.max_pressure)
    x_span = view_max - view_min
    if x_span < 1e-9:
      x_span = 1.0
    x_padding = max(x_span * 0.05, 1.0)
    max_voltage = self.effective_max_voltage()
    self.plot_widget.setXRange(view_min - x_padding, view_max + x_padding, padding=0.0)
    self.plot_widget.setYRange(0.0, 5.0, padding=0.05)
    self.graph_points.set_max_voltage(max_voltage)

    self.curve_line.setData(x_values, y_values)
    self.graph_points.set_points(self.curve_points)

  def on_graph_point_moved(self, index, x_value, y_value):
    self.curve_points[index][0] = round(x_value, 2)
    self.curve_points[index][1] = round(y_value, 2)
    self.expand_pressure_range_to_points()
    self.populate_table()
    self._set_selected_row(index)
    self.refresh_graph()

  def on_graph_drag_state_changed(self, index, dragging):
    if dragging and 0 <= index < len(self.curve_points):
      self._drag_start_state = self._capture_state()
      self._set_selected_row(index)
      return

    if not dragging and self._drag_start_state is not None:
      end_state = self._capture_state()
      if not self._states_equal(self._drag_start_state, end_state):
        self._record_history_state()
      self._drag_start_state = None

  def _set_selected_row(self, row):
    if row < 0 or row >= self.points_table.rowCount():
      return
    if self.points_table.currentRow() == row:
      return
    self.points_table.selectRow(row)

  def show_voltage_limit_popup(self):
    QtWidgets.QMessageBox.information(
      self,
      "Scale Limit",
      f"Voltage is capped at {self.effective_max_voltage():.2f}V by the current scale. Increase scale to allow higher voltage output."
    )

  def show_pressure_limit_popup(self):
    QtWidgets.QMessageBox.information(
      self,
      "Pressure Limit",
      "Pressure movement is capped by neighboring points to keep the curve ordered. Move adjacent points first to extend this point's range."
    )

  def save_configuration(self):
    payload = {
      "points": self.curve_points,
      "scale": self.scale,
      "min_pressure": self.min_pressure,
      "max_pressure": self.max_pressure
    }
    self.set_status("Configuration prepared")
    # Serial protocol hookup can be finalized later.
    # self.serial_worker.send_line("SET_CONFIG:" + json.dumps(payload))

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
