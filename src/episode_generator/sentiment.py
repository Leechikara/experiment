# coding = utf-8
import random


class Sentiment(object):
    def __init__(self, script):
        self.script = script

    def episode_generator(self, polarity, attr):
        episode_script = list()

        scene_name = ' '.join([str(polarity), str(attr)])
        scene_content = list()

        available_script = self.script[polarity][attr]
        for turn in available_script:
            if type(turn) == list:
                scene_content.append(random.choice(turn))
            else:
                scene_content.append(turn)

        scene = {scene_name: scene_content}
        episode_script.append(scene)

        return episode_script
