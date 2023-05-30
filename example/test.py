#!/usr/bin/env python

import importlib.util

spec = importlib.util.spec_from_file_location('user',"analysis.py")
user = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user)

print(user.SAMPLES_DEF)


