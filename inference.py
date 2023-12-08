# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference demo for YAMNet."""
from __future__ import division, print_function

import sys
import threading

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

import params
import yamnet as yamnet_model
import time
import streamlit as st
import pandas as pd


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

def process_audio(file_name):
    graph = tf.Graph()
    with graph.as_default():
      yamnet = yamnet_model.yamnet_frames_model(params)
      yamnet.load_weights('yamnet.h5')
    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')
  
    # Decode the WAV file.
    print(f"Processing {file_name}...")
    wav_data, sr = sf.read(file_name, dtype=np.int16, samplerate=16000, channels=1, subtype='PCM_16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != params.SAMPLE_RATE:
      waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

    # Predict YAMNet classes.
    # Second output is log-mel-spectrogram array (used for visualizations).
    # (steps=1 is a work around for Keras batching limitations.)
    with graph.as_default():
      scores, _ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
    # Scores is a matrix of (time_frames, num_classes) classifier scores.
    # Average them along time to get an overall classifier output for the clip.
    prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    top10_i = np.argsort(prediction)[::-1][:10]
    output =  ':\n' + '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i]) for i in top10_i)
    print(file_name, output)
    timestamp = str(int(time.time()))
    filename = f"g:\\temp\{timestamp}.txt"
    with open(filename, 'w') as file:
    # Write two lines to the file
      file.write(file_name)
      file.write(output)

class Watcher:

    def __init__(self, directory=".", handler=FileSystemEventHandler()):
        self.observer = Observer()
        self.handler = handler
        self.directory = directory
       
    def start(self):
        self.observer.schedule(self.handler, self.directory, recursive=False)
        self.observer.start()

    def stop(self):
        self.observer.stop()

    def join(self):
        self.observer.join()    

    # def run(self):
    #     self.observer.schedule(
    #         self.handler, self.directory, recursive=True)
    #     self.observer.start()
    #     print("\nWatcher Running in {}/\n".format(self.directory))
    #     try:
    #         while True:
    #             time.sleep(1)
    #     except:
    #         self.observer.stop()
    #     self.observer.join()
    #     print("\nWatcher Terminated\n")
              
        
result_filename = ""

def process_result(filename):
  global result_filename
  result_filename=filename
        
class SoundFileHandler(FileSystemEventHandler):

    def on_any_event(self, event):
      if event.event_type == "created":
          process_audio(event.src_path)
            
class ResultFileHandler(FileSystemEventHandler):

    def on_any_event(self, event):
      if event.event_type == "created":
          process_result(event.src_path)
            
            

def printer_thread():
  global result_filename
  while True:
    time.sleep(1)
    if result_filename != "":
      print("printing")
      st.write(result_filename)
      result_filename = ""
    

if __name__ == '__main__':
  
  result_filename = "fister"

  t = Thread(target=printer_thread)
  add_script_run_ctx(t)
  t.start()
  
  w1 = Watcher("G:\\temp\\322717bbfdd19e262e6c2c9ad8296e2c", SoundFileHandler())
  w2 = Watcher("g:\\temp", ResultFileHandler())

  w2.start()
  w1.start()

  try:
    while True:
      time.sleep(5)
  except:
    w1.stop()
    w2.stop()
    print("error")
    
  w1.join()
  w2.join()
  t.join()
  
  process_audio(sys.argv[1:])
