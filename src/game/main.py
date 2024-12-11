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
#from test import SingleObjectTrackingPipeline

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]
PLAYER_HEALTH = 1
INIT_SCORE = 999


# --------------------------------------------------------------------------
# -- player class
# --------------------------------------------------------------------------
class Player(pygame.sprite.Sprite):
    # -----------------------------------------------------------
    # init class

    def __init__(self,name):
        super(Player, self).__init__()
        self.surf = pygame.Surface((0, 0), pygame.SRCALPHA)  # Hier muss die Bounding Box rein
        self.rect = self.surf.get_rect()
        # start at screen center
        self.image = self.surf
        self.health = PLAYER_HEALTH
        self.score = INIT_SCORE
        self.name = name

    # -----------------------------------------------------------
    # update player rectangle
    def update_box_position(self, x, y, w, h):

        self.rect.update(x, y, w, h)
        self.surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(self.surf, (0, 255, 0), (0, 0, w, h), 6)  # Grünen Rahmen zeichnen
        self.image = self.surf

    def update(self, pipeline):

        if len(pipeline.tracker.box_tracks) > 0:
            for track in pipeline.tracker.box_tracks:
                x, y, w, h = track["box"]
                self.update_box_position(x, y, w, h)

    def catch_fruit(self, fruit):
        self.score += fruit.base_value + int(fruit.base_value * fruit.multiplier)
        fruit.kill()

    def catch_bomb(self, bomb):
        bomb.kill()


# --------------------------------------------------------------------------
# -- game
# --------------------------------------------------------------------------

# init pygame
pygame.init()

# set display size and caption
#screen = pygame.display.set_mode(SCREEN)
pygame.display.set_caption("Computer Vision Game")

# init game clock
fps = 30
clock = pygame.time.Clock()

# opencv - init webcam capture
cap = cv.VideoCapture("../../assets/images/DS-Default-Pulli-Hell-LR.mov")
# set width & height to screen size
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen.get_width())
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen.get_height())
start_time = None
screen = pygame.display.set_mode([cap.get(cv.CAP_PROP_FRAME_WIDTH),cap.get(cv.CAP_PROP_FRAME_HEIGHT)])

def start_game():
    global start_time

    start_time = pygame.time.get_ticks()


def stop_game(player): # Den Spieler übergeben der Verloren hat: für allein oder zweispieler
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

    global start_time
    start_game()

    player1 = Player('Player1')  # TODO Player Parameter Egal weil ja detektion außer name
    player2 = Player('Player2')

    players = pygame.sprite.Group(player1)

    fruits = pygame.sprite.Group()
    bombs = pygame.sprite.Group()

    generator = Generator.ObjectGenerator(
        initial_fruit_interval=4.5,
        initial_bomb_interval=5.5,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT
    )

    #pipeline = SingleObjectTrackingPipeline(cap)

    running = True

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
        imgRGB = cv.cvtColor(cameraFrame, cv.COLOR_BGR2RGB)
        # image needs to be rotated for pygame
        imgRGB = np.rot90(imgRGB)
        # convert image to pygame and visualize
        gameFrame = pygame.surfarray.make_surface(imgRGB).convert_alpha()

        screen.blit(gameFrame, (0, 0))

        #pipeline.run(cameraFrame)
        # --------------------------------------------------------------
        #keys = pygame.key.get_pressed()
        #for player in players:
        #    player.update(keys)

        #player1.update(pipeline)

        # Früchte und Bomben generator
        new_fruit = generator.generate_fruit()
        if new_fruit:
            fruits.add(new_fruit)

        new_bomb = generator.generate_bomb(random.choice(list(players)))
        if new_bomb:
            bombs.add(new_bomb)

        # Früchte und Bomben bewegen updaten etc.
        fruits.update()
        bombs.update()

        # Frucht und Spieler
        for fruit in fruits:
            if pygame.sprite.spritecollide(fruit, players, False):
                player1.catch_fruit(fruit)

        for bomb in bombs:
            if pygame.sprite.spritecollide(bomb, players, False):
                player1.catch_bomb(bomb)
                 # Spieler 1 verliert bei Bombe

        for player in players:
            if player.health <= 0:
                pass
                #stop_game(player)

        # Früchte und Bomben zeichnen
        fruits.draw(screen)
        bombs.draw(screen)
        players.draw(screen)

        # render score & hp
        Overlay.draw_score(screen, players)
        Overlay.draw_hp(screen, players)
        Overlay.draw_game_time(screen, start_time)

        # update  screen
        pygame.display.flip()
        # set clock
        clock.tick(fps)

    # quit game
    pygame.quit()

    # release capture
    cap.release()


if __name__ == "__main__":
    main()
