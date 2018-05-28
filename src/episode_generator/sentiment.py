# coding = utf-8
import random


class Sentiment(object):
    def __init__(self, script):
        self.script = script
        self.episode_script = None

    def init_episode(self):
        self.episode_script = list()

    def episode_generator(self, polarity, attr):
        self.init_episode()

        scene_name = ' '.join([str(polarity), str(attr)])
        scene_content = list()

        available_script = self.script[polarity][attr]
        for turn in available_script:
            if type(turn) == list:
                scene_content.append(random.choice(turn))
            else:
                scene_content.append(turn)

        scene = {scene_name: scene_content}
        self.episode_script.append(scene)

        return self.episode_script
