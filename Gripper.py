"""
Author: Briana Bouchard
Gripper control via stepper motor on Raspberry Pi 4 (L298N driver).
Uses RPi.GPIO. Run with: sudo python3 Gripper.py

TUNING:
  - Adjust CLOSE_STEPS to control how far the gripper closes at pickup
  - Adjust OPEN_STEPS to control how far the gripper opens at dropoff
  - Adjust STEP_DELAY to control motor speed (smaller = faster, min ~0.002)
"""

import RPi.GPIO as GPIO
import time

# --- PIN CONFIGURATION (BOARD numbering, same as original) ---
OUT1 = 7
OUT2 = 11
OUT3 = 13
OUT4 = 15

# --- TUNING PARAMETERS ---
CLOSE_STEPS = 40   # Steps to close gripper at pickup  <-- adjust this
OPEN_STEPS  = 40   # Steps to open gripper at dropoff  <-- adjust this
STEP_DELAY  = 0.03  # Delay between steps in seconds    <-- adjust for speed

# --- SETUP ---
GPIO.setmode(GPIO.BOARD)
for pin in [OUT1, OUT2, OUT3, OUT4]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Step sequence
STEP_SEQUENCE = [
    (GPIO.HIGH, GPIO.LOW,  GPIO.HIGH, GPIO.LOW),
    (GPIO.LOW,  GPIO.HIGH, GPIO.HIGH, GPIO.LOW),
    (GPIO.LOW,  GPIO.HIGH, GPIO.LOW,  GPIO.HIGH),
    (GPIO.HIGH, GPIO.LOW,  GPIO.LOW,  GPIO.HIGH),
]

def _move(steps, direction=1):
    """
    Internal: move stepper a given number of steps.
    direction=1  -> close (forward)
    direction=-1 -> open  (reverse)
    """
    current_step = 0
    for _ in range(steps):
        s = STEP_SEQUENCE[current_step % 4]
        GPIO.output(OUT1, s[0])
        GPIO.output(OUT2, s[1])
        GPIO.output(OUT3, s[2])
        GPIO.output(OUT4, s[3])
        time.sleep(STEP_DELAY)
        current_step = (current_step + direction) % 4

    # Power down coils after move to prevent heat buildup
    for pin in [OUT1, OUT2, OUT3, OUT4]:
        GPIO.output(pin, GPIO.LOW)


def gripper_open(steps=None):
    """Close the gripper (call at pickup point)."""
    n = steps if steps is not None else CLOSE_STEPS
    print(f"Gripper closing ({n} steps)...")
    _move(n, direction=1)
    print("Gripper closed.")


def gripper_close(steps=None):
    """Open the gripper (call at dropoff point)."""
    n = steps if steps is not None else OPEN_STEPS
    print(f"Gripper opening ({n} steps)...")
    _move(n, direction=-1)
    print("Gripper open.")


def gripper_cleanup():
    """Call this when your program exits to release GPIO pins."""
    GPIO.cleanup()


# --- STANDALONE TEST ---
if __name__ == "__main__":
    try:
        print("=== Gripper Test ===")
        print("Closing...")
        gripper_close()
        time.sleep(5)
        print("Opening...")
        gripper_open()
        time.sleep(1)
        print("Done! Adjust CLOSE_STEPS / OPEN_STEPS at the top of the file.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        gripper_cleanup()