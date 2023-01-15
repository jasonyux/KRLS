"""
   MTTOD: utils/io_utils.py

   implements simple I/O utilities for serialized objects and
   logger definitions.

   Copyright 2021 ETRI LIRS, Yohan Lee. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import json
import pickle
import logging


def save_json(obj, save_path, indent=4):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(load_path, lower=True):
    with open(load_path, "r", encoding="utf-8") as f:
        obj = f.read()

        if lower:
            obj = obj.lower()

        return json.loads(obj)


def save_pickle(obj, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(load_path):
    with open(load_path, "rb") as f:
        return pickle.load(f)


def save_text(obj, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for o in obj:
            f.write(o + "\n")


def load_text(load_path, lower=True):
    with open(load_path, "r", encoding="utf-8") as f:
        text = f.read()
        if lower:
            text = text.lower()
        return text.splitlines()
