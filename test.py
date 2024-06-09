import serial
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates

# Set up the serial connection to the Arduino
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # Wait for the serial connection to initialize

# Initialize plot
fig, ax1 = plt.subplots()

# Create secondary y-axis
ax2 = ax1.twinx()

# Initialize data lists
x_data = []
y_value_data = []
y_level_data = []

line_value, = ax1.plot_date(x_data, y_value_data, '-', label='Vibration Value', color='blue')
line_level, = ax2.plot_date(x_data, y_level_data, '-', label='Vibration Level', color='red')

# Formatting the x-axis to show current time
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xticks(rotation=45)
ax1.set_xlabel('Time')
ax1.set_ylabel('Vibration Value', color='blue')
ax2.set_ylabel('Vibration Level', color='red')
plt.title('Real-time Vibration Sensor Data')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

# Function to categorize vibration value
def categorize_vibration(value):
    value = int(value)
    if 0 <= value <= 1000:
        return 0
    elif 1001 <= value <= 10000:
        return 1
    else:
        return 2
# Function to read vibration value from Arduino and save to csv
def read_vibration():
    if ser.in_waiting > 0:
        vibration_value = ser.readline().decode('utf-8').strip()
        vibration_level = categorize_vibration(vibration_value)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Vibration value: {vibration_value}, Vibration level: {vibration_level}")
        writer.writerow([timestamp, vibration_value, vibration_level])
        return datetime.now(), int(vibration_value), vibration_level
    return None, None, None

# Function to update the plot
def update(frame):
    timestamp, vibration_value, vibration_level = read_vibration()
    if timestamp and vibration_value is not None:
        x_data.append(timestamp)
        y_value_data.append(vibration_value)
        y_level_data.append(vibration_level)

        line_value.set_data(x_data, y_value_data)
        line_level.set_data(x_data, y_level_data)

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
    return line_value, line_level

# Open a .csv file to append the readings
with open('VibRead_test.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    # Check if the file is empty to write the header
    file.seek(0, 2)  # Move the cursor to the end of the file
    if file.tell() == 0:
        writer.writerow(['Timestamp', 'Vibration Value', 'Vibration Level'])  # Write the header row if the file is empty

    # Set up the plot animation
    ani = FuncAnimation(fig, update, interval=50)

    try:
        # Show the plot
        plt.show()
        # Infinite loop to continuously check the vibration sensor
        while True:
            read_vibration()
            time.sleep(0.05)  # Delay of 50 milliseconds
    except KeyboardInterrupt:
        print("Exiting program")
    finally:
        ser.close()  # Close the serial connection



