import cv2
import time
import random
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to track time and punches
prev_frame = None
punch_count = 0
last_punch_time = 0  # Timestamp of the last detected punch
punch_interval = 0.5  # Minimum time interval (in seconds) between punches
calories_per_punch = 0.3  # Calories burned per punch

# Health tips and boxing move explanations
tips = [
    "Remember to take breaks and stay hydrated!",
    "A jab is a quick, straight punch with your lead hand.",
    "A cross is a powerful, straight punch with your rear hand.",
    "An uppercut is a rising punch aimed at the opponent's chin.",
    "A hook is a punch delivered in a circular motion with your lead hand.",
    "Maintain proper form to avoid injuries!",
    "Stretch before and after your workout for flexibility."
]

# Punch rank classification
def classify_punch_rank(punch_count):
    if punch_count >= 30:  # Tyson-level punches
        return "Tyson", (0, 0, 255)  # Red for Tyson
    elif 20 <= punch_count < 30:  # Muhammad Ali-level punches
        return "Muhammad Ali", (255, 215, 0)  # Gold for Ali
    elif 10 <= punch_count < 20:  # Pro-level punches
        return "Pro", (0, 255, 0)  # Green for Pro
    elif 5 <= punch_count < 10:  # Amateur-level punches
        return "Amateur", (0, 255, 255)  # Yellow for Amateur
    else:  # Fewer than 5 punches
        return "Beginner", (255, 255, 255)  # White for Beginner

# Collect user weight and age for calorie calculation
print("Enter your details for personalized calorie calculation.")
try:
    weight = float(input("Enter your weight (kg): "))
    age = int(input("Enter your age: "))
except ValueError:
    print("Invalid input! Using default values.")
    weight = 70
    age = 25

# Adjust calorie calculation
calories_per_punch = 0.3 * (weight / 70)

# Main loop variables
start_time = time.time()
last_tip_time = start_time
current_tip = ""
tip_start_time = None

def should_display_tip():
    return current_tip != "" and time.time() - tip_start_time <= 4

def draw_graph(frame, count):
    graph_width = 400
    graph_height = 100
    graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
    bar_width = int((count / 30) * graph_width)  # Normalize to a max of 30 punches
    bar_color = (0, 255, 0)
    cv2.rectangle(graph, (0, 0), (bar_width, graph_height), bar_color, -1)
    cv2.putText(graph, f"Punches: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    frame[480:580, 200:600] = graph

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for better spacing
    frame = cv2.resize(frame, (800, 600))

    # Convert frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize previous frame
    if prev_frame is None:
        prev_frame = gray
        continue

    # Compute the absolute difference between current and previous frame
    frame_diff = cv2.absdiff(prev_frame, gray)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each detected contour
    punch_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Ignore small movements
            punch_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Increment punch count only if sufficient time has passed since the last punch
    current_time = time.time()
    if punch_detected and (current_time - last_punch_time) > punch_interval:
        punch_count += 1
        last_punch_time = current_time

    # Calculate calories burned
    calories_burned = punch_count * calories_per_punch

    # Classify punch rank
    punch_rank, rank_color = classify_punch_rank(punch_count)

    # Display punch details on the screen
    cv2.putText(frame, f"Punch Count: {punch_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Calories Burned: {calories_burned:.1f} cal", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Punch Rank: {punch_rank}", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, rank_color, 2)

    # Display elapsed time
    elapsed_time = int(current_time - start_time)
    cv2.putText(frame, f"Time: {elapsed_time}s", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Update health tips every 20 seconds
    if current_time - last_tip_time > 20:
        current_tip = random.choice(tips)
        last_tip_time = current_time
        tip_start_time = current_time

    # Display current tip if within 4 seconds
    if should_display_tip():
        cv2.putText(frame, current_tip, (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the graph
    draw_graph(frame, punch_count)

    # Display instructions
    cv2.putText(frame, "Press 'q' to quit", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Punch Detection", frame)

    # Update the previous frame
    prev_frame = gray

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Display session summary
def display_summary():
    print("\n=== Workout Summary ===")
    print(f"Total Punches: {punch_count}")
    print(f"Calories Burned: {calories_burned:.1f} cal")
    print(f"Final Rank: {punch_rank}")

display_summary()

# Update leaderboard
try:
    with open("leaderboard.txt", "a") as f:
        f.write(f"{punch_count}\n")
    with open("leaderboard.txt", "r") as f:
        scores = [int(line.strip()) for line in f.readlines()]
        scores.sort(reverse=True)
        print("\n=== Leaderboard ===")
        for i, score in enumerate(scores[:5], start=1):  # Top 5 scores
            print(f"{i}. {score} punches")
except Exception as e:
    print("Error handling leaderboard:", e)

# Release resources
cap.release()
cv2.destroyAllWindows()
