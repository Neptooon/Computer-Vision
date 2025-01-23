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


class Player(pygame.sprite.Sprite):
    """
        Repräsentiert einen Spieler im Spiel. Jeder Spieler ist mit einer Bounding Box verbunden,
        die durch die Tracking-Daten aktualisiert wird.
    """
    # init class
    def __init__(self,name):
        """
        Initialisiert den Spieler

        Args:
            name (str): name.
        """
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

        """
        updated die Position der Box des jeweiligen Spielers

        Args:
            track (Track): Track Objekt
        """
        if not track.lost:
            x, y, w, h = track.box
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            self.rect.update(x, y, w, h)
            self.surf = pygame.Surface((w, h), pygame.SRCALPHA)
            if track.id == 1:
                pygame.draw.rect(self.surf, (0, 0, 255), (0, 0, w, h), 6)  # Blauen Rahmen zeichnen
            else:
                pygame.draw.rect(self.surf, (255, 0, 0), (0, 0, w, h), 6)  # Roten Rahmen zeichnen
            font = pygame.font.Font(None, 36)
            text = font.render(str(track.id), True, (0, 0, 0))
            self.surf.blit(text, (5, 5))

            self.image = self.surf
        else:
            # BoundingBox entfernen
            self.rect.update(0, 0, 0, 0)
            self.surf = pygame.Surface((0, 0), pygame.SRCALPHA)
            self.image = self.surf

    # updated den Spieler mit der jeweiligen Box des zugehörigen Tracks
    def update(self, track):
        """
        Übergibt den Track

        Args:
            track (Track): Track Objekt
        """
        if track.box is not None:
            self.update_box_position(track)

    # Spieler catched frucht
    def catch_fruit(self, fruit):
        """
        Berechnet den Wert der Frucht und schreibt sie dem Score des jeweiligen Spielers gut

        Args:
            fruit (Fruit): Fruit Objekt
        """

        self.score += fruit.base_value + int(fruit.base_value * fruit.multiplier)
        fruit.kill()

    # Spieler catched bombe
    def catch_bomb(self, bomb):
        """
        Löscht gefangene Bombe

        Args:
            bomb (Bomb): Bomben Objekt
        """
        bomb.kill()


# --------------------------------------------------------------------------
# -- game
# --------------------------------------------------------------------------

# init pygame
pygame.init()

# setzt display size, caption & init time
pygame.display.set_caption("Computer Vision Game")
fps = 60
clock = pygame.time.Clock()

# opencv - init webcam capture & set width & height
cap = cv.VideoCapture(0)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

screen = pygame.display.set_mode([SCREEN_WIDTH,SCREEN_HEIGHT])

cap.set(cv.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)


def start_game():  # Startet Spiel-Timer
    return pygame.time.get_ticks()


def stop_game(players):  # Stoppt das Spiel oder Restarted es
    Overlay.draw_game_over(screen, players) # Zeichnet Game Over Overlay

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() # Quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                if event.key == pygame.K_r:
                    main()  # Neustart
                    waiting = False


def main():

    Overlay.draw_start_screen(screen)  # Zeichnet Start Screen
    Overlay.draw_countdown(screen)  # Zeichnet countdown

    start_time = start_game()  # Startet das Game

    # Spieler initialisieren
    player1 = Player('Player1')
    player2 = Player('Player2')
    player_list = [player1, player2]


    # Gruppen für Spieler, Früchte und Bomben
    players = pygame.sprite.Group(player1, player2)
    fruits = pygame.sprite.Group()
    bombs = pygame.sprite.Group()

    # Generator für Früchte und Bomben
    generator = Generator.ObjectGenerator(
        initial_fruit_interval=4.5,
        initial_bomb_interval=5.5,
        screen_width=int(SCREEN_WIDTH),
        screen_height=int(SCREEN_HEIGHT)
    )

    # Tracking-Pipeline initialisieren
    pipeline = MultipleObjectTrackingPipeline(cap)

    running = True

    # Hauptspiel-Loop
    while running:
        # Ereignisverarbeitung
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

        imgRGB = np.rot90(imgRGB) # Bild drehen

        gameFrame = pygame.surfarray.make_surface(imgRGB).convert_alpha() # In pygame-Format konvertieren

        screen.blit(gameFrame, (0, 0))

        # Spieler mit Tracking-Daten aktualisieren
        for track in pipeline.tracker.tracks:
            if track.id == 1:
                player1.update(track)  # Spieler Updaten
                player1.activated = True
            elif track.id == 2:
                player2.update(track)  # Spieler Updaten
                player2.activated = True


        # Früchte und Bomben generieren
        new_fruit = generator.generate_fruit()
        if new_fruit:
            fruits.add(new_fruit)

        player_choice = random.choice(list(players))
        new_bomb = generator.generate_bomb(player_choice if player_choice.activated else None)
        if new_bomb is not None:
            bombs.add(new_bomb)

        # Früchte und Bomben bewegen updaten etc.
        fruits.update()
        bombs.update()

        # Kollisionen überprüfen
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
                stop_game(player_list)


        # Objekte zeichnen
        fruits.draw(screen)
        bombs.draw(screen)
        players.draw(screen)

        # Overlay-Elemente (Score, Zeit, Gesundheit) anzeigen
        Overlay.draw_score(screen, players)
        Overlay.draw_hp(screen, players)
        Overlay.draw_game_time(screen, start_time)

        # Bildschirm aktualisieren
        pygame.display.flip()

        # framerate
        clock.tick(fps)

    # quit
    pygame.quit()
    cap.release()


if __name__ == "__main__":
    main()
