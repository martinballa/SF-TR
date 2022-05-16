import os
import numpy as np
import pygame
from pygame.locals import *

class Renderer():
    def __init__(self, assets_path, size=(600, 600), fps=10):
        pygame.init()
        self.fps = fps
        ntiles = 12 # todo obs size
        self.tile_size = size[0] // ntiles
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption('MiniCrafter')

        self.tiles = {}

        # read in assets
        for asset in os.listdir(assets_path):
            tile = pygame.image.load(os.path.join(assets_path, asset))
            tile = pygame.transform.scale(tile, (self.tile_size, self.tile_size))
            self.tiles[f"{asset.split('.')[0]}"] = tile

        self.clock = pygame.time.Clock()

    def render(self, obs):
        map = obs
        for x in range(map.shape[0]):
            for y in range(map.shape[1]):
                # if map[x][y] == 0:
                self.screen.blit(self.tiles["grass"], (self.tile_size * x, self.tile_size * y))
                if map[x][y] == 1:
                    self.screen.blit(self.tiles["tree"], (self.tile_size * x, self.tile_size * y))
                if map[x][y] == 2:
                    self.screen.blit(self.tiles["iron"], (self.tile_size * x, self.tile_size * y))
                if map[x][y] == 3:
                    self.screen.blit(self.tiles["coal"], (self.tile_size * x, self.tile_size * y))
                if map[x][y] == 4:
                    self.screen.blit(self.tiles["water"], (self.tile_size * x, self.tile_size * y))
                if map[x][y] == 5:
                    self.screen.blit(self.tiles["table"], (self.tile_size * x, self.tile_size * y))
                    print("crafting table")
                if map[x][y] == 6:
                    self.screen.blit(self.tiles["player"], (self.tile_size * x, self.tile_size * y))
        pygame.display.flip()
        self.clock.tick(10)

if __name__ == "__main__":
    # dummy test
    renderer = Renderer(os.path.expanduser("./assets"))
    obs = np.array([[4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [4, 4, 0, 0, 0, 0, 1, 0, 3, 2, 4, 0],
                    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0],
                    [0, 1, 1, 4, 0, 0, 2, 0, 0, 0, 0, 0],
                    [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    renderer.render(obs)

    pygame.display.flip()
    running = True
    while (running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()

