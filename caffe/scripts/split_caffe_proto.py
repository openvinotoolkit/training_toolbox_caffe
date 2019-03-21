# pylint: skip-file

# 
# All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#!/usr/bin/env python
import mmap
import re
import os
import errno

script_path = os.path.dirname(os.path.realpath(__file__))

# a regex to match the parameter definitions in caffe.proto
r = re.compile(r'(?://.*\n)*message ([^ ]*) \{\n(?: .*\n|\n)*\}')

# create directory to put caffe.proto fragments
try:
    os.mkdir(
        os.path.join(script_path,
                     '../docs/_includes/'))
    os.mkdir(
        os.path.join(script_path,
                     '../docs/_includes/proto/'))
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

caffe_proto_fn = os.path.join(
    script_path,
    '../src/caffe/proto/caffe.proto')

with open(caffe_proto_fn, 'r') as fin:

    for m in r.finditer(fin.read(), re.MULTILINE):
        fn = os.path.join(
            script_path,
            '../docs/_includes/proto/%s.txt' % m.group(1))
        with open(fn, 'w') as fout:
            fout.write(m.group(0))
