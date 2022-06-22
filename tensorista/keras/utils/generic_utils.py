import numpy as np
import time
import sys

if sys.platform == "ios":
    #from objc_util import *
    import objc_util


class Progbar:
    def __init__(self,
                 target,
                 width=30,
                 verbose=1,
                 interval=0.05,
                 stateful_metrics=None,
                 unit_name='step'):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self._seen_so_far = 0


    def update(self, current, values=None, finalize=None):
        # if target==None:     200/Unknown - 2s 10ms/step - acc: 0.4862 - pr: 0.6632Model: "seq1"
        # print("50/200 [======>.......................] - ETA: 4s - acc: 0.5953 - pr: 0.6362")
        #              [==============================]
        """verbose_steps = int(iterations // 30)
            if i % verbose_steps == 0:
                iteration = ((len(str(iterations))-len(str(i))) * " ") + str(i) + "/" + str(iterations)
                loading_bar = "[" + (i//verbose_steps)*"=" + (30-(i//verbose_steps))*" " + "]"
                sys.stdout.write(iteration + loading_bar)
                sys.stdout.write('\r')
                #print(
                #    iteration,
                #    loading_bar,
                #    "- 0s 0us/step -",
                #    "loss:", round(cross_entropy_loss(Y, A2), 4),
                #    "- acc: 0.0000",
                #    end="\r"
                #)"""
        self._seen_so_far = current
        if sys.platform == "ios":
            textview = objc_util.UIApplication.sharedApplication().keyWindow().rootViewController().accessoryViewController().consoleViewController().view().subviews()[0]
            textstorage = textview.textStorage()
            end_char = textstorage.length()
            r = textstorage.paragraphRangeForCharacterRange_(objc_util.NSRange(end_char-1))
            p = textstorage.paragraphs()[r.location-1]
            textstorage.replaceCharactersInRange_withString_(p.range(),objc_util.ns(''))
            textview.setNeedsLayout()
            endline = '\n'
        else:
            endline = "\r"
        print(
            str(self._seen_so_far) + "/" +
            str(self.target) + " [" +
            "=" * int(self._seen_so_far // (self.target/self.width)+1) +
            "." * (self.width - (int(
                np.ceil(self._seen_so_far // (self.target/self.width)))+1)) +
            "] - " +
            "ETA: ",
            values, end=endline)


    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


    def _estimate_step_duration(self, current, now):
        print(time.time())
