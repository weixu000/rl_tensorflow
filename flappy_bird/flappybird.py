import pygame
import sys
import random
import numpy as np
from PIL import Image
import os


class FlappyBird:
    def __init__(self):
        self.screen = pygame.display.set_mode((400, 700))
        self.bird = pygame.Rect(65, 200, 50, 50)
        self.background = pygame.image.load(os.path.dirname(__file__) + "/assets/background.png").convert()
        self.birdSprites = [pygame.image.load(os.path.dirname(__file__) + "/assets/1.png").convert_alpha(),
                            pygame.image.load(os.path.dirname(__file__) + "/assets/2.png").convert_alpha(),
                            pygame.image.load(os.path.dirname(__file__) + "/assets/dead.png")]
        self.wallUp = pygame.image.load(os.path.dirname(__file__) + "/assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load(os.path.dirname(__file__) + "/assets/top.png").convert_alpha()
        self.gap = 150
        self.wallx = 400
        self.birdY = 200
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 5
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-110, 110)
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 50)

    def updateWalls(self):
        self.wallx -= 2
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-110, 110)

    def updateBird(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())
        if upRect.colliderect(self.bird) or downRect.colliderect(self.bird):
            self.dead = True
        if not 0 < self.bird[1] < 650:
            self.dead = True

    def restart(self):
        self.bird[1] = random.randint(200, 500)
        self.birdY = self.bird[1]
        self.dead = False
        self.counter = 0
        self.wallx = 400
        self.offset = random.randint(-110, 110)
        self.gravity = 5

    def jumpAction(self):
        self.jump = 17
        self.gravity = 5
        self.jumpSpeed = 10

    def update(self, render_human=True):
        self.screen.fill((0, 0, 0))
        if render_human:
            self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.wallUp,
                         (self.wallx, 360 + self.gap - self.offset))
        self.screen.blit(self.wallDown,
                         (self.wallx, 0 - self.gap - self.offset))
        if render_human:
            self.screen.blit(self.font.render(str(self.counter), -1, (255, 255, 255)), (200, 50))
        if self.dead:
            self.sprite = 2
        elif self.jump:
            self.sprite = 1
        self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
        if not self.dead:
            self.sprite = 0
        self.updateWalls()
        self.updateBird()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                    self.jumpAction()

            self.update()
            if self.dead and not 0 < self.bird[1] < 720: self.restart()
            pygame.display.update()


class FlappyBirdEnv:
    def __init__(self):
        self.action_n = 2
        self.observation_shape = [80, 80]
        self.game = FlappyBird()

    def reset(self):
        self.game.restart()
        self.game.update(False)
        return self.get_observation()

    def render(self):
        pygame.display.update()

    def step(self, action):
        pygame.event.get()  # 出队所有事件，防止卡死
        if self.game.dead: raise Exception('Flappy bird is dead and need reset')
        if action == 1: self.game.jumpAction()
        for _ in range(2): self.game.update(False)
        observation = self.get_observation()
        done = self.game.dead
        reward = -56.0 if done else 1.0
        return observation, reward, done, None

    def get_observation(self):
        image = Image.fromarray(pygame.surfarray.array3d(self.game.screen).transpose((1, 0, 2)))
        image = image.convert('L').resize(self.observation_shape)
        image = np.array(image)
        return image


if __name__ == "__main__":
    FlappyBird().run()
