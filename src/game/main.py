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
from src.cv_modules.main import MultipleObjectTrackingPipeline


#SCREEN_WIDTH = 1280
#SCREEN_WIDTH = cv.CAP_PROP_FRAME_WIDTH
#SCREEN_HEIGHT = 720
#SCREEN_HEIGHT = cv.CAP_PROP_FRAME_HEIGHT

class Player(pygame.sprite.Sprite):
    # init class
    def __init__(self,name):
        super(Player, self).__init__()
        self.surf = pygame.Surface((0, 0), pygame.SRCALPHA)
        self.rect = self.surf.get_rect()
        # start at screen center
        self.image = self.surf
        self.health = 1
        self.score = 0
        self.name = name
        self.activated = False


    # updated die bbox
    def update_box_position(self, track):
        x, y, w, h = track.box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        self.rect.update(x, y, w, h)
        self.surf = pygame.Surface((w, h), pygame.SRCALPHA)
        if track.id == 1:
            pygame.draw.rect(self.surf, (0, 0, 255), (0, 0, w, h), 6)  # Grünen Rahmen zeichnen
        else:
            pygame.draw.rect(self.surf, (255, 0, 0), (0, 0, w, h), 6)  # Grünen Rahmen zeichnen
        font = pygame.font.Font(None, 36)
        text = font.render(str(track.id), True, (0, 0, 0))


        self.surf.blit(text, (5,5))

        self.image = self.surf

    # Checked die tracks und leitet die bbox weiter
    def update(self, track):
        if track.box is not None:
            self.update_box_position(track)


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
fps = 60
clock = pygame.time.Clock()

# opencv - init webcam capture & set width & height
cap = cv.VideoCapture('../../assets/videos/MOT-Livedemo1.mov')
SCREEN_WIDTH = cap.get(cv.CAP_PROP_FRAME_WIDTH)
SCREEN_HEIGHT = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
screen = pygame.display.set_mode([SCREEN_WIDTH,SCREEN_HEIGHT])

#cap.set(cv.CAP_PROP_FRAME_WIDTH, screen.get_width())
#cap.set(cv.CAP_PROP_FRAME_HEIGHT, screen.get_height())


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
                    main()
                    waiting = False


def main():

    Overlay.draw_start_screen(screen)
    Overlay.draw_countdown(screen)

    start_time = start_game()


    player1 = Player('Player1')
    player2 = Player('Player2')


    #Gruppen
    players = pygame.sprite.Group(player1, player2)
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
    pipeline = MultipleObjectTrackingPipeline(cap)

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
        cameraFrame = pipeline.run(cameraFrame) # Tracking Starten

        imgRGB = cv.cvtColor(cameraFrame, cv.COLOR_BGR2RGB)
        # image needs to be rotated for pygame
        imgRGB = np.rot90(imgRGB)
        # convert image to pygame and visualize
        gameFrame = pygame.surfarray.make_surface(imgRGB).convert_alpha()

        screen.blit(gameFrame, (0, 0))

        for track in pipeline.tracker.tracks:
            if track.id == 1 and not track.lost:
                player1.update(track)  # Spieler Updaten
                player1.activated = True
            elif track.id == 2 and not track.lost:
                player2.update(track)  # Spieler Updaten
                player2.activated = True


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
            if player1.rect.colliderect(fruit.rect):
                player1.catch_fruit(fruit)
            if player2.rect.colliderect(fruit.rect):
                player2.catch_fruit(fruit)

        for bomb in bombs:
            if player1.rect.colliderect(bomb.rect) and bomb.bomb_type == 'blue':
                player1.catch_bomb(bomb)
            if player2.rect.colliderect(bomb.rect) and bomb.bomb_type == 'red':
                player2.catch_bomb(bomb)

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
