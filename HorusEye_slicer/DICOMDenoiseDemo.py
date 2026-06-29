"""Minimal 3D Slicer demo for single-slice DICOM CT denoising."""

from __future__ import annotations

import json
import os
import subprocess
import traceback
from pathlib import Path

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


MODULE_DIR = Path(__file__).resolve().parent
RUNNER_PATH = MODULE_DIR / "denoise_inference.py"


class DICOMDenoiseDemo(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "DICOM Denoise Demo"
        self.parent.categories = ["Denoising"]
        self.parent.dependencies = []
        self.parent.contributors = ["DICOM Denoise Demo contributors"]
        self.parent.helpText = "Minimal single-slice DICOM denoising demo with configurable window width and level."
        self.parent.acknowledgementText = "Uses a SwinUNETR-based denoising inference example."


class DICOMDenoiseDemoWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = DICOMDenoiseDemoLogic()

        parameters_collapsible_button = ctk.ctkCollapsibleButton()
        parameters_collapsible_button.text = "Parameters"
        self.layout.addWidget(parameters_collapsible_button)
        form_layout = qt.QFormLayout(parameters_collapsible_button)

        self.dicom_path_edit = self._path_row(
            form_layout,
            "Input DICOM:",
            "",
            "Select one DICOM CT image file.",
            file_mode=True,
        )
        self.python_path_edit = self._path_row(
            form_layout,
            "External Python:",
            os.environ.get("DICOM_DENOISE_PYTHON", os.environ.get("HIPAS_PYTHON", "")),
            "Select the Python executable from the environment with torch, MONAI, pydicom, and Pillow.",
            file_mode=True,
        )
        self.checkpoint_path_edit = self._path_row(
            form_layout,
            "Checkpoint:",
            "",
            "Select HorusEye_demo.pth or a compatible checkpoint. Not required for smoke test.",
            file_mode=True,
        )
        self.output_dir_edit = self._path_row(
            form_layout,
            "Output directory:",
            str(Path(slicer.app.temporaryPath) / "DICOMDenoiseDemo"),
            "Directory where output PNG, NPY, and metadata files will be written.",
            file_mode=False,
        )

        self.window_width_spin = qt.QDoubleSpinBox()
        self.window_width_spin.minimum = 1.0
        self.window_width_spin.maximum = 10000.0
        self.window_width_spin.value = 400.0
        self.window_width_spin.decimals = 1
        form_layout.addRow("Window width:", self.window_width_spin)

        self.window_level_spin = qt.QDoubleSpinBox()
        self.window_level_spin.minimum = -5000.0
        self.window_level_spin.maximum = 5000.0
        self.window_level_spin.value = 40.0
        self.window_level_spin.decimals = 1
        form_layout.addRow("Window level:", self.window_level_spin)

        self.smoke_test_checkbox = qt.QCheckBox()
        self.smoke_test_checkbox.checked = False
        self.smoke_test_checkbox.toolTip = "Skip the neural network and run a lightweight DICOM/windowing/output test."
        form_layout.addRow("Smoke test:", self.smoke_test_checkbox)

        self.apply_button = qt.QPushButton("Run")
        self.apply_button.toolTip = "Run DICOM denoising with the configured external Python environment."
        form_layout.addRow(self.apply_button)

        self.log_text = qt.QPlainTextEdit()
        self.log_text.readOnly = True
        self.log_text.minimumHeight = 180
        form_layout.addRow("Log:", self.log_text)

        self.apply_button.connect("clicked(bool)", self.on_apply_button)
        self.layout.addStretch(1)

    def _path_row(self, form_layout, label, initial_path, tooltip, file_mode):
        widget = qt.QWidget()
        row = qt.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        line_edit = qt.QLineEdit()
        line_edit.text = initial_path
        line_edit.toolTip = tooltip
        browse_button = qt.QPushButton("Browse")
        browse_button.toolTip = tooltip
        row.addWidget(line_edit)
        row.addWidget(browse_button)
        form_layout.addRow(label, widget)

        def browse():
            if file_mode:
                selected = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), "Select file", line_edit.text)
            else:
                selected = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Select directory", line_edit.text)
            if selected:
                line_edit.text = selected

        browse_button.connect("clicked(bool)", browse)
        return line_edit

    def append_log(self, text):
        self.log_text.appendPlainText(text.rstrip())
        slicer.app.processEvents()

    def on_apply_button(self):
        self.apply_button.enabled = False
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            result = self.logic.run(
                dicom_path=Path(self.dicom_path_edit.text.strip()),
                python_path=Path(self.python_path_edit.text.strip()),
                checkpoint_path=Path(self.checkpoint_path_edit.text.strip()) if self.checkpoint_path_edit.text.strip() else None,
                output_dir=Path(self.output_dir_edit.text.strip()),
                window_width=self.window_width_spin.value,
                window_level=self.window_level_spin.value,
                smoke_test=self.smoke_test_checkbox.checked,
                log_callback=self.append_log,
            )
            self.append_log("Done.")
            self.append_log(json.dumps(result, indent=2))
        except Exception as exc:
            self.append_log("ERROR: " + str(exc))
            self.append_log(traceback.format_exc())
            slicer.util.errorDisplay(str(exc), windowTitle="DICOM Denoise Demo")
        finally:
            qt.QApplication.restoreOverrideCursor()
            self.apply_button.enabled = True


class DICOMDenoiseDemoLogic(ScriptedLoadableModuleLogic):
    def run(
        self,
        dicom_path,
        python_path,
        checkpoint_path,
        output_dir,
        window_width,
        window_level,
        smoke_test=False,
        log_callback=None,
    ):
        if not dicom_path.is_file():
            raise FileNotFoundError(f"Input DICOM was not found: {dicom_path}")
        if not python_path.is_file():
            raise FileNotFoundError(f"External Python was not found: {python_path}")
        if not RUNNER_PATH.is_file():
            raise FileNotFoundError(f"Denoise runner was not found: {RUNNER_PATH}")
        if not smoke_test and (checkpoint_path is None or not checkpoint_path.is_file()):
            raise FileNotFoundError("Select a checkpoint, or enable smoke test.")

        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(python_path),
            str(RUNNER_PATH),
            "--input-dicom",
            str(dicom_path),
            "--output-dir",
            str(output_dir),
            "--window-width",
            str(float(window_width)),
            "--window-level",
            str(float(window_level)),
        ]
        if smoke_test:
            command.append("--smoke-test")
        else:
            command.extend(["--checkpoint", str(checkpoint_path)])

        if log_callback:
            log_callback("Running: " + subprocess.list2cmdline(command))
        completed = subprocess.run(command, capture_output=True, text=True, cwd=str(MODULE_DIR))
        if completed.stdout and log_callback:
            log_callback(completed.stdout)
        if completed.stderr and log_callback:
            log_callback(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(f"Denoising failed with exit code {completed.returncode}.")

        metadata_path = output_dir / "denoise_outputs.json"
        result = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.load_denoised_volume(output_dir / "denoised.npy", result)
        return result

    def load_denoised_volume(self, npy_path, metadata):
        import numpy as np

        image = np.load(str(npy_path)).astype(np.float32)
        volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Denoised DICOM slice")
        slicer.util.updateVolumeFromArray(volume_node, image[np.newaxis, :, :])

        spacing = metadata.get("dicom", {}).get("pixel_spacing", [1.0, 1.0])
        if len(spacing) >= 2:
            volume_node.SetSpacing(float(spacing[1]), float(spacing[0]), 1.0)

        display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeDisplayNode")
        display_node.SetAutoWindowLevel(False)
        display_node.SetWindowLevel(1.0, 0.5)
        volume_node.SetAndObserveDisplayNodeID(display_node.GetID())
        slicer.util.setSliceViewerLayers(background=volume_node, fit=True)


class DICOMDenoiseDemoTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.delayDisplay("DICOM Denoise Demo module loaded.")
