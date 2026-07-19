"""Reusable themed Tkinter widgets used by the desktop application."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class LogConsole(ttk.Frame):
    """Scrollable logging console that is safe to update from the UI thread."""

    def __init__(self, master: tk.Misc, **kwargs: object) -> None:
        super().__init__(master, **kwargs)
        self.text = tk.Text(self, height=9, wrap="word", state="disabled", bg="#17202A", fg="#EAECEE", insertbackground="white", relief="flat")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def write(self, message: str) -> None:
        """Append a timestamp-free operational message."""
        self.text.configure(state="normal")
        self.text.insert("end", message.rstrip() + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")


class StatusBar(ttk.Frame):
    """Small status surface shared by all navigation views."""

    def __init__(self, master: tk.Misc, **kwargs: object) -> None:
        super().__init__(master, **kwargs)
        self.variable = tk.StringVar(value="Ready")
        self.label = ttk.Label(self, textvariable=self.variable, anchor="w")
        self.label.pack(fill="x", padx=10, pady=5)

    def set(self, message: str) -> None:
        """Update status text."""
        self.variable.set(message)
