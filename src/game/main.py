"""
-----------------------------------------------------------------------------
Vorlesung: Computer Vision (Wintersemester 2024/25)
Thema: pygame example with integrated OpenCV

-----------------------------------------------------------------------------
"""

import random

import cv2 as cv
import numpy as np
import pygame
import generator as Generator
import overlay as Overlay
from game_tracker import SingleObjectTrackingPipeline


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]


class Player(pygame.sprite.Sprite):
    # init class
    def __init__(self,name):
        super(Player, self).__init__()
        self.surf = pygame.Surface((0, 0), pygame.SRCALPHA)
        self.rect = self.surf.get_rect()
        # start at screen center
        self.image = self.surf
        self.health = 4
        self.score = 0
        self.name = name


    # updated die bbox
    def update_box_position(self, x, y, w, h):

        self.rect.update(x, y, w, h)
        self.surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(self.surf, (255, 0, 0), (0, 0, w, h), 6)  # Grünen Rahmen zeichnen
        self.image = self.surf


    # Checked die tracks und leitet die bbox weiter
    def update(self, pipeline):

        if len(pipeline.tracker.box_tracks) > 0:
            for track in pipeline.tracker.box_tracks:
                x, y, w, h = track["box"]
                self.update_box_position(x, y, w, h)

    # Spieler catched frucht
    def catch_fruit(self, fruit):
        self.score += fruit.base_value + int(fruit.base_value * fruit.multiplier)
        fruit.kill()

    # Spieler catched bombe
    def catch_bomb(self, bomb):
        bomb.kill()


# --------------------------------------------------------------------------
# -- game
# --------------------------------------------------------------------------

# init pygame
pygame.init()

# set display size, caption & init time
pygame.display.set_caption("Computer Vision Game")
fps = 30
clock = pygame.time.Clock()

# opencv - init webcam capture & set width & height
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
screen = pygame.display.set_mode([SCREEN_WIDTH,SCREEN_HEIGHT])

def start_game(): # Startet Spieltimer

    return pygame.time.get_ticks()


def stop_game(player): # Stop Game / Game Over für den übergebenen Spieler
    Overlay.draw_game_over(screen)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() # Quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                if event.key == pygame.K_r:
                    main()  # Restart
                    waiting = False


def main():

    start_time = start_game()

    player1 = Player('Player1')  # Spieler Anzahl erstmal auf 1 da SOT: Todo in MOT Spieler dynamisch erstellen
    #player2 = Player('Player2')


    #Gruppen
    players = pygame.sprite.Group(player1)
    fruits = pygame.sprite.Group()
    bombs = pygame.sprite.Group()

    # Früchte und Bomben Generator
    generator = Generator.ObjectGenerator(
        initial_fruit_interval=4.5,
        initial_bomb_interval=5.5,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT
    )

    #Tracking Pipeline
    pipeline = SingleObjectTrackingPipeline(cap)

    running = True

    #Main loop
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # press 'esc' to quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # -- opencv & viz image
        ret, cameraFrame = cap.read()
        if cameraFrame is None:
            break
        cameraFrame = pipeline.tracker_run(cameraFrame) # Tracking Starten

        imgRGB = cv.cvtColor(cameraFrame, cv.COLOR_BGR2RGB)
        # image needs to be rotated for pygame
        imgRGB = np.rot90(imgRGB)
        # convert image to pygame and visualize
        gameFrame = pygame.surfarray.make_surface(imgRGB).convert_alpha()

        screen.blit(gameFrame, (0, 0))

        player1.update(pipeline)  # Spieler Updaten

        # Früchte und Bomben generieren
        new_fruit = generator.generate_fruit()
        if new_fruit:
            fruits.add(new_fruit)

        new_bomb = generator.generate_bomb(random.choice(list(players)))
        if new_bomb:
            bombs.add(new_bomb)

        # Früchte und Bomben bewegen updaten etc.
        fruits.update()
        bombs.update()

        # Frucht/Bomben Kollision
        for fruit in fruits:
            if pygame.sprite.spritecollide(fruit, players, False):
                player1.catch_fruit(fruit)

        for bomb in bombs:
            if pygame.sprite.spritecollide(bomb, players, False):
                player1.catch_bomb(bomb)

        # Game Over Check
        for player in players:
            if player.health <= 0:
                stop_game(player)

        # Früchte und Bomben zeichnen
        fruits.draw(screen)
        bombs.draw(screen)
        players.draw(screen)

        # rendern score & hp
        Overlay.draw_score(screen, players)
        Overlay.draw_hp(screen, players)
        Overlay.draw_game_time(screen, start_time)

        # screen updaten
        pygame.display.flip()

        # uhr
        clock.tick(fps)

    # quit
    pygame.quit()
    cap.release()


if __name__ == "__main__":
    main()
