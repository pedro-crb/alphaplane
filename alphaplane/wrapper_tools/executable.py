from __future__ import annotations

import subprocess
import threading
import tempfile
import time
import os
import io
import re

from typing import Self


class Executable:
    def __init__(self, executable_path: str, capture_output: bool = False,
                 timeout_seconds: float = 10.0) -> None:
        self.executable_path = executable_path
        self.process = None
        self.temp_dir = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.capture_output = capture_output
        self.output_buffer = []
        self.timeout = timeout_seconds
        self.start_time = None
        self.last_response = None
        self.buffer_condition = threading.Condition()

    def read_stream(self, stream: io.TextIOWrapper, capture_output: bool, output_buffer, buffer_condition):
        for line in iter(stream.readline, ''):
            if capture_output:
                with buffer_condition:
                    output_buffer.append(line)
                    buffer_condition.notify()
                    self.last_response = time.time()

    def start_process(self) -> None:
        if self.process is None:
            self.temp_dir = tempfile.mkdtemp()
            self.process = subprocess.Popen(
                [self.executable_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0, universal_newlines=True, cwd=self.temp_dir)

            self.start_time = time.time()
            self.stdout_thread = threading.Thread(target=self.read_stream,
                                                  args=(self.process.stdout, self.capture_output,
                                                        self.output_buffer, self.buffer_condition))
            self.stderr_thread = threading.Thread(target=self.read_stream,
                                                  args=(self.process.stderr, False, None, None))
            self.stdout_thread.start()
            self.stderr_thread.start()
            if self.timeout is not None:
                timeout_thread = threading.Thread(target=self.monitor_timeout)
                timeout_thread.start()

    def monitor_timeout(self) -> None:
        while self.process and self.process.poll() is None:
            if self.last_response is None:
                self.last_response = self.start_time

            assert self.last_response is not None
            elapsed_time = time.time() - self.last_response
            if elapsed_time > self.timeout:
                self.process.terminate()
                self.process.wait()
                break
            time.sleep(0.1)

    def wait_for_output(self, wait_for):
        pattern = re.compile('|'.join(map(str, map(re.escape, wait_for))))
        response = None
        with self.buffer_condition:
            while self.process and self.process.poll() is None:
                if any(pattern.search(line) for line in self.output_buffer):
                    response = ''.join(self.output_buffer)
                    break
                if not self.buffer_condition.wait(timeout=0.1):
                    response = None
                    break
            return response

    def send_command(self, command: str, wait_for=None, clear_buffer=True):
        response = None
        if self.process and self.process.poll() is None:
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()
            with self.buffer_condition:
                self.buffer_condition.wait(timeout=0.01)
            if wait_for:
                response = self.wait_for_output(wait_for)
            if clear_buffer:
                self.clear_buffer()
            return response

    def clear_buffer(self):
        with self.buffer_condition:
            self.output_buffer.clear()

    def cleanup(self) -> None:
        try:
            if self.process:
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()

                if self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait()
        except Exception as e:
            print(f"Error during subprocess cleanup: {e}")

        try:
            if self.stdout_thread and self.stdout_thread.is_alive():
                with self.buffer_condition:
                    self.buffer_condition.notify_all()
                self.stdout_thread.join()

            if self.stderr_thread and self.stderr_thread.is_alive():
                self.stderr_thread.join()
        except Exception as e:
            print(f"Error during thread cleanup: {e}")

        try:
            assert self.temp_dir is not None
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, file))
                os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"Error during temp directory cleanup: {e}")

    def __enter__(self) -> Self:
        self.start_process()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
