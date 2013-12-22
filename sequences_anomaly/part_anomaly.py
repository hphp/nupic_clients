#!/usr/bin/env python
# ----------------------------------------------------------------------
# hp_carrot
# Copyright (C) 2013,
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# www.baidu.com/p/hp_carrot
# ----------------------------------------------------------------------


"""A simple client to read number and predict it in real time."""

from collections import deque
import time

import psutil
import matplotlib.pyplot as plt

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory

import model_params
print model_params.MODEL_PARAMS
steps = int(model_params.MODEL_PARAMS['modelParams']['clParams']['steps'])
#exit(0)
SECONDS_PER_STEP = 0.002
WINDOW = 60

# turn matplotlib interactive mode on (ion)
'''
plt.ion()
fig = plt.figure()
# plot title, legend, etc
plt.title('Number prediction example')
plt.xlabel('time [s]')
plt.ylabel('CPU usage [%]')
'''
import random
import copy

def generate(last_number, GenerateBlank=False):
    if GenerateBlank:
        if random.randrange(10) == 3:
            return -1
    return ( last_number + 200 ) % 3000

def runPredNumber():
  """Poll CPU usage, make predictions, and plot the results. Runs forever."""
  # Create the model for predicting CPU usage.
  model = ModelFactory.create(model_params.MODEL_PARAMS)
  model.enableInference({'predictedField': 'number'})
  # The shifter will align prediction and actual values.
  shifter = InferenceShifter()
  # Keep the last WINDOW predicted and actual values for plotting.
  actHistory = deque([0.0] * WINDOW, maxlen=60)
  predHistory = deque([0.0] * WINDOW, maxlen=60)

  # Initialize the plot lines that we will update with each new record.
  actline, = plt.plot(range(WINDOW), actHistory)
  predline, = plt.plot(range(WINDOW), predHistory)
  # Set the y-axis range.
  actline.axes.set_ylim(0, 3000)
  predline.axes.set_ylim(0, 3000)

  last_number = 0

  for step in range(130):
    s = time.time()

    # Get the current number.
    number = generate(last_number)
    last_number = number

    # Run the input through the model and shift the resulting prediction.
    modelInput = {'number': number}
    true_result = model.run(modelInput)
    if step % 10 == 0:
        print ' --------------------------- '
        model.resetSequenceStates()

        print model.getRuntimeStats()

    anomalyScore = true_result.inferences['anomalyScore']
    print("Anomaly detected at [%s]. Anomaly score: %f.",
                  true_result.rawInput["number"], anomalyScore)

    result = shifter.shift(true_result)

    # Update the trailing predicted and actual value deques.
    inference = result.inferences['multiStepBestPredictions'][steps]

    print step,result.inferences['multiStepBestPredictions'] , modelInput['number']

    if inference is not None:
      actHistory.append(result.rawInput['number'])
      predHistory.append(inference)

    # Redraw the chart with the new data.
    '''
    actline.set_ydata(actHistory)  # update the data
    predline.set_ydata(predHistory)  # update the data
    plt.draw()
    plt.legend( ('actual','predicted') )    
    # Make sure we wait a total of 2 seconds per iteration.
    try: 
      plt.pause(SECONDS_PER_STEP)
    except:
      pass
    '''

#print dir(model)
  model.finishLearning()
  #model.disableLearning()
  print model.isLearningEnabled()
  print "finish learning"
  last_prediction = 0
  old_model = copy.deepcopy(model)
  for part_number in range(10):
      last_number = random.randrange(15) * 200
      #model.resetSequenceStates()
      for step in range(10):
        s = time.time()

        # Get the current number.
        number = generate(last_number)
        #number = generate(last_number, GenerateBlank=True)
        last_number = number

        '''
        if random.randrange(10) == 3:
            number = random.randrange(3000)
            print "this input is not correct", number
        '''
        '''
        if number < 0:
            print "this input is blank", number
            number = int(last_prediction)
        '''

        # Run the input through the model and shift the resulting prediction.
        modelInput = {'number': number}
        result = (model.run(modelInput))

        true_result = result
        anomalyScore = true_result.inferences['anomalyScore']
        print("Anomaly detected at [%s]. Anomaly score: %f.",
                      true_result.rawInput["number"], anomalyScore)
        # Update the trailing predicted and actual value deques.
        inference = result.inferences['multiStepBestPredictions'][steps]
        print result.inferences['multiStepBestPredictions'] , modelInput['number'] #,result.inferences #,",prediction: ", result.inferences['prediction']
        last_prediction = inference

        if inference is not None:
          actHistory.append(result.rawInput['number'])
          predHistory.append(inference)

        # Redraw the chart with the new data.
        '''
        actline.set_ydata(actHistory)  # update the data
        predline.set_ydata(predHistory)  # update the data
        plt.draw()
        plt.legend( ('actual','predicted') )    

        # Make sure we wait a total of 2 seconds per iteration.
        try: 
          plt.pause(SECONDS_PER_STEP)
        except:
          pass
        '''
      model = copy.deepcopy(old_model)

if __name__ == "__main__":
  runPredNumber()
