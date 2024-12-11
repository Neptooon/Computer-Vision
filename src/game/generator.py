import objects as Objects
import random
import time


class ObjectGenerator:
    def __init__(self, initial_fruit_interval=5.0, initial_bomb_interval=6.0,
                 min_fruit_interval=2.0, min_bomb_interval=3.0,
                 speedup_rate=0.95, update_interval=7.0, screen_width=1280, screen_height=720):

        self.fruit_interval = initial_fruit_interval  # Startintervall: Zeit in Sek. die vergeht um eine Frucht zu spawnen
        self.bomb_interval = initial_bomb_interval  # Startintervall: Zeit in Sek. die vergeht um eine Bombe zu spawnen
        self.min_fruit_interval = min_fruit_interval  # Minimales Intervall für Früchte damit es nicht zu schnell wird
        self.min_bomb_interval = min_bomb_interval  # Minimales Intervall für Bomben damit es nicht zu schnell wird
        self.speedup_rate = speedup_rate  # Änderungsrate welche die Spawnzetien verringert (0.99 verringert die Intervallzeiten um 1% pro update)
        self.update_interval = update_interval  # Zeit in Sek die vergeht, um ein Intervall update durchzuführen
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Letzte Objekt Spawn-Zeiten und Updates der Intervalle
        self.last_fruit_spawn = time.time()
        self.last_bomb_spawn = time.time()
        self.last_update_time = time.time()

        self.last_fruit_increase = time.time()  # Zeit in Sek. der letzten erhöhung der Früchte
        self.fruit_number = 1  # Min Anzahl Früchte

    def update_spawn_intervals(self):
        # Intervall Zeiten der Objekte verkürzen, um das Spiel schneller & dynamischer zu machen
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.fruit_interval = max(self.min_fruit_interval, self.fruit_interval * self.speedup_rate)
            self.bomb_interval = max(self.min_bomb_interval, self.bomb_interval * self.speedup_rate)
            # Zeit des letzten Updates festhalten
            self.last_update_time = current_time

    def increase_fruit_spawn(self):
        # Erhöht die Anzahl der Früchte alle X Sekunden
        current_time = time.time()
        if current_time - self.last_fruit_increase >= 10:  # Alle X Sekunden eine Frucht hinzufügen (max.2)

            if self.fruit_number < 2:  # Max 2 Früchte Spawnen möglich
                self.fruit_number += 1
            # Zeit der letzten Ink. aktualisieren
            self.last_fruit_increase = current_time

    def generate_fruit(self):

        self.update_spawn_intervals()  # Spawn-Zeiten updaten
        self.increase_fruit_spawn()  # ggf. Fruchtanzahl erhöhen

        current_time = time.time()  # Früchte Spawnen lassen, wenn das Intervall abgelaufen ist
        if current_time - self.last_fruit_spawn >= self.fruit_interval:
            self.last_fruit_spawn = current_time

            fruits = [
                Objects.Fruit(
                    random.randint(0, self.screen_width), 0,
                    random.choice(list(Objects.FRUIT_SPRITES.keys())),
                    random.uniform(1.0, 10.0), random.randint(5, 20),
                    depth=random.uniform(0.6, 0.9)
                )
                for _ in range(self.fruit_number)
            ]  # Früchte Spawn
            return fruits
        return None

    def generate_bomb(self, player):
        # Bomben spielerspezifisch generieren
        self.update_spawn_intervals()  # Spawn-Zeiten update

        current_time = time.time()  # Bomben Spawnen lassen, wenn das Intervall abgelaufen ist
        if current_time - self.last_bomb_spawn >= self.bomb_interval:
            self.last_bomb_spawn = current_time

            depth = random.uniform(0.5, 0.7)
            bomb_type = 'blue' if player.name == 'Player1' else 'red'
            bomb = Objects.Bomb(
                random.randint(0, self.screen_width), 0,
                bomb_type, player,
                random.uniform(5, 20), random.uniform(2.0, 10.0), depth=depth
            )  # Bomben Spawn
            return bomb
        return None
