"""Tkinter GUI for Audio Enhancer."""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from .core import AudioReconstructor
from .gpu import GPUBackend, detect_gpu
from .pipeline import PipelineConfig, StageConfig


class AudioEnhancerGUI:
    """Main GUI application for Audio Enhancer."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio Enhancer")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)

        self.input_files: list[Path] = []
        self.output_dir: Optional[Path] = None
        self.processing = False
        self.message_queue: queue.Queue = queue.Queue()

        self.gpu_info = detect_gpu()

        self._create_widgets()
        self._create_menu()

        self._process_messages()

    def _create_menu(self) -> None:
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Add Files...", command=self._add_files)
        file_menu.add_command(
            label="Set Output Directory...", command=self._set_output_dir
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        gpu_frame = ttk.LabelFrame(main_frame, text="GPU Status", padding="5")
        gpu_frame.pack(fill=tk.X, pady=(0, 10))

        gpu_text = f"{self.gpu_info.device_name}"
        if self.gpu_info.backend != GPUBackend.CPU:
            gpu_text += f" ({self.gpu_info.vram_gb:.1f} GB VRAM)"
        gpu_label = ttk.Label(gpu_frame, text=gpu_text)
        gpu_label.pack(side=tk.LEFT)

        backend_label = ttk.Label(
            gpu_frame,
            text=f"Backend: {self.gpu_info.backend.value.upper()}",
            foreground="green" if self.gpu_info.backend != GPUBackend.CPU else "gray",
        )
        backend_label.pack(side=tk.RIGHT)

        input_frame = ttk.LabelFrame(main_frame, text="Input Files", padding="5")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        list_frame = ttk.Frame(input_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=6)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(btn_frame, text="Add Files...", command=self._add_files).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(
            btn_frame, text="Remove Selected", command=self._remove_selected
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear All", command=self._clear_files).pack(
            side=tk.LEFT
        )

        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.pack(fill=tk.X, pady=(0, 10))

        self.output_var = tk.StringVar(value="Same as input")
        ttk.Entry(output_frame, textvariable=self.output_var, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5)
        )
        ttk.Button(output_frame, text="Browse...", command=self._set_output_dir).pack(
            side=tk.RIGHT
        )

        stages_frame = ttk.LabelFrame(main_frame, text="Pipeline Stages", padding="5")
        stages_frame.pack(fill=tk.X, pady=(0, 10))

        self.stage_vars = {}
        stages = [
            ("denoise", "Denoising (for noisy recordings)", False),
            ("super_resolution", "Super Resolution (recover compression loss)", True),
            ("harmonic", "Harmonic Enhancement", False),
            ("mastering", "Final Mastering (limiter + dither)", True),
            ("normalize", "Loudness Normalization (-14 LUFS)", False),
        ]

        for i, (key, label, default) in enumerate(stages):
            var = tk.BooleanVar(value=default)
            self.stage_vars[key] = var
            cb = ttk.Checkbutton(stages_frame, text=label, variable=var)
            cb.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=10, pady=2)

        format_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="5")
        format_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(format_frame, text="Format:").grid(row=0, column=0, padx=(0, 5))
        self.format_var = tk.StringVar(value="flac")
        format_combo = ttk.Combobox(
            format_frame,
            textvariable=self.format_var,
            values=["flac", "wav", "ogg", "opus"],
            state="readonly",
            width=10,
        )
        format_combo.grid(row=0, column=1, padx=(0, 20))

        ttk.Label(format_frame, text="Sample Rate:").grid(row=0, column=2, padx=(0, 5))
        self.sample_rate_var = tk.StringVar(value="48000")
        sr_combo = ttk.Combobox(
            format_frame,
            textvariable=self.sample_rate_var,
            values=["44100", "48000", "96000"],
            state="readonly",
            width=10,
        )
        sr_combo.grid(row=0, column=3, padx=(0, 20))

        ttk.Label(format_frame, text="Bit Depth:").grid(row=0, column=4, padx=(0, 5))
        self.bit_depth_var = tk.StringVar(value="24")
        bd_combo = ttk.Combobox(
            format_frame,
            textvariable=self.bit_depth_var,
            values=["16", "24", "32"],
            state="readonly",
            width=10,
        )
        bd_combo.grid(row=0, column=5)

        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor=tk.W)

        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.log_text = tk.Text(log_frame, height=8, state="disabled", wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        log_scroll = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scroll.set)

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X)

        self.process_btn = ttk.Button(
            action_frame,
            text="Start Processing",
            command=self._start_processing,
            style="Accent.TButton",
        )
        self.process_btn.pack(side=tk.RIGHT, padx=(5, 0))

        self.cancel_btn = ttk.Button(
            action_frame,
            text="Cancel",
            command=self._cancel_processing,
            state="disabled",
        )
        self.cancel_btn.pack(side=tk.RIGHT)

    def _add_files(self) -> None:
        """Add audio files to the queue."""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.ogg *.opus *.webm *.m4a *.aac"),
            ("Video files", "*.mp4 *.mkv *.avi *.webm"),
            ("All files", "*.*"),
        ]
        files = filedialog.askopenfilenames(filetypes=filetypes)
        for f in files:
            path = Path(f)
            if path not in self.input_files:
                self.input_files.append(path)
                self.file_listbox.insert(tk.END, path.name)

    def _remove_selected(self) -> None:
        """Remove selected files from the queue."""
        selection = self.file_listbox.curselection()
        for i in reversed(selection):
            self.file_listbox.delete(i)
            del self.input_files[i]

    def _clear_files(self) -> None:
        """Clear all files from the queue."""
        self.file_listbox.delete(0, tk.END)
        self.input_files.clear()

    def _set_output_dir(self) -> None:
        """Set output directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir = Path(directory)
            self.output_var.set(str(self.output_dir))

    def _log(self, message: str) -> None:
        """Add message to log."""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def _update_progress(self, progress: float, stage: str, message: str) -> None:
        """Update progress bar and status."""
        self.progress_var.set(progress * 100)
        self.status_var.set(f"{stage}: {message}")

    def _process_messages(self) -> None:
        """Process messages from worker thread."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                msg_type = msg.get("type")

                if msg_type == "log":
                    self._log(msg["message"])
                elif msg_type == "progress":
                    self._update_progress(msg["progress"], msg["stage"], msg["message"])
                elif msg_type == "done":
                    self._processing_complete(msg.get("error"))
        except queue.Empty:
            pass

        self.root.after(100, self._process_messages)

    def _build_config(self) -> PipelineConfig:
        """Build pipeline config from UI state."""
        return PipelineConfig(
            extraction=StageConfig(
                enabled=True,  # Always enabled - required to load audio
                params={
                    "sample_rate": int(self.sample_rate_var.get()),
                    "normalize": True,
                },
            ),
            denoise=StageConfig(
                enabled=self.stage_vars["denoise"].get(),
                params={"use_resemble_enhance": True},
            ),
            super_resolution=StageConfig(
                enabled=self.stage_vars["super_resolution"].get(),
                params={"model": "basic", "ddim_steps": 50},
            ),
            harmonic_enhancement=StageConfig(
                enabled=self.stage_vars["harmonic"].get(),
                params={"harmonic_boost_db": 3.0, "stereo_width": 1.2},
            ),
            final_mastering=StageConfig(
                enabled=self.stage_vars["mastering"].get(),
                params={
                    "normalize_loudness": self.stage_vars["normalize"].get(),
                    "target_lufs": -14.0,
                    "true_peak_dbtp": -1.0,
                    "output_bit_depth": int(self.bit_depth_var.get()),
                },
            ),
            output_format=self.format_var.get(),
            output_sample_rate=int(self.sample_rate_var.get()),
            output_bit_depth=int(self.bit_depth_var.get()),
        )

    def _start_processing(self) -> None:
        """Start processing in background thread."""
        if not self.input_files:
            messagebox.showwarning("No Files", "Please add audio files to process.")
            return

        self.processing = True
        self.process_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress_var.set(0)

        # Start worker thread
        thread = threading.Thread(target=self._process_files, daemon=True)
        thread.start()

    def _process_files(self) -> None:
        """Process files in background thread."""
        try:
            config = self._build_config()
            reconstructor = AudioReconstructor(config, self.gpu_info)

            total_files = len(self.input_files)

            for i, input_file in enumerate(self.input_files):
                if not self.processing:
                    break

                self.message_queue.put(
                    {
                        "type": "log",
                        "message": f"Processing {input_file.name} ({i+1}/{total_files})",
                    }
                )

                if self.output_dir:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = (
                        self.output_dir
                        / f"{input_file.stem}_restored.{config.output_format}"
                    )
                else:
                    output_path = (
                        input_file.parent
                        / f"{input_file.stem}_restored.{config.output_format}"
                    )

                def progress_callback(
                    progress: float, stage: str, message: str
                ) -> None:
                    overall = (i + progress) / max(total_files, 1)
                    self.message_queue.put(
                        {
                            "type": "progress",
                            "progress": overall,
                            "stage": stage,
                            "message": message,
                        }
                    )

                reconstructor.process(input_file, output_path, progress_callback)

                self.message_queue.put(
                    {"type": "log", "message": f"Saved: {output_path}"}
                )

            reconstructor.unload_models()
            self.message_queue.put({"type": "done"})

        except Exception as e:
            self.message_queue.put({"type": "done", "error": str(e)})

    def _cancel_processing(self) -> None:
        """Cancel current processing."""
        self.processing = False
        self.status_var.set("Cancelling...")

    def _processing_complete(self, error: Optional[str] = None) -> None:
        """Called when processing is complete."""
        self.processing = False
        self.process_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")

        if error:
            self.status_var.set(f"Error: {error}")
            self._log(f"Error: {error}")
            messagebox.showerror("Error", error)
        else:
            self.status_var.set("Complete")
            self._log("Processing complete!")
            messagebox.showinfo("Complete", "All files processed successfully!")

    def _show_about(self) -> None:
        """Show about dialog."""
        messagebox.showinfo(
            "About Audio Reconstructor",
            "Audio Reconstructor v0.1.0\n\n"
            "Advanced audio restoration pipeline with:\n"
            "- DSP-based bandwidth extension\n"
            "- AI denoising (noisereduce)\n"
            "- Harmonic enhancement\n"
            "- Professional mastering\n\n"
            "Open-source audio restoration for everyone.",
        )

    def run(self) -> None:
        """Run the GUI application."""
        self.root.mainloop()


def main() -> None:
    """Main entry point for GUI."""
    app = AudioEnhancerGUI()
    app.run()


if __name__ == "__main__":
    main()
