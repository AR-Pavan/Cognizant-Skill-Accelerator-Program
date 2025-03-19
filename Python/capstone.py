import datetime

# --------------------------- Task Management ---------------------------

class Task:
    def init(self, name, deadline, priority, time_estimate):
        self.name = name
        self.deadline = deadline
        self.priority = priority
        self.time_estimate = time_estimate

    def str(self):
        return f"Task: {self.name}, Deadline: {self.deadline}, Priority: {self.priority}, Estimated Time: {self.time_estimate} hours"

# Function to get a valid task deadline input
def get_valid_deadline():
    while True:
        try:
            deadline_str = input("Enter task deadline (YYYY-MM-DD): ")
            deadline = datetime.datetime.strptime(deadline_str, "%Y-%m-%d")
            return deadline
        except ValueError:
            print("Invalid date format. Please enter the date in the format YYYY-MM-DD.")

# Function to get a valid priority input
def get_valid_priority():
    while True:
        try:
            priority = int(input("Enter task priority (1 = highest, 5 = lowest): "))
            if 1 <= priority <= 5:
                return priority
            else:
                print("Priority must be between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter an integer between 1 and 5.")

# Function to get a valid time estimate input
def get_valid_time_estimate():
    while True:
        try:
            time_estimate = float(input("Enter estimated time to complete (in hours): "))
            if time_estimate > 0:
                return time_estimate
            else:
                print("Time estimate must be positive.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to input tasks
def input_task():
    name = input("Enter task name: ")
    deadline = get_valid_deadline()
    priority = get_valid_priority()
    time_estimate = get_valid_time_estimate()
    return Task(name, deadline, priority, time_estimate)

# Function to display tasks sorted by deadline and priority
def display_tasks(tasks):
    tasks = sorted(tasks, key=lambda x: (x.deadline, x.priority))
    print("\nYour Task List (Sorted by Deadline and Priority):")
    for task in tasks:
        print(task)

# ------------------------ Performance Tracking ------------------------

class Subject:
    def init(self, name):
        self.name = name
        self.scores = []

    def add_score(self, score):
        self.scores.append(score)

    def average_score(self):
        return sum(self.scores) / len(self.scores) if self.scores else 0

    def str(self):
        return f"Subject: {self.name}, Average Score: {self.average_score():.2f}"

# Function to get a valid score input
def get_valid_score():
    while True:
        try:
            score = float(input("Enter score: "))
            if 0 <= score <= 100:
                return score
            else:
                print("Score must be between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a valid score between 0 and 100.")

# Function to input scores for a subject
def input_scores_for_subject(subject):
    while True:
        try:
            score = get_valid_score()
            subject.add_score(score)
            more_scores = input("Would you like to add another score? (y/n): ").lower()
            if more_scores != 'y':
                break
        except Exception as e:
            print(f"An error occurred: {e}")

# Function to display performance across subjects
def display_performance(subjects):
    print("\nPerformance Summary:")
    for subject in subjects:
        print(subject)
    print("\nIdentifying Weak Areas (Average below 60):")
    for subject in subjects:
        if subject.average_score() < 60:
            print(f"Need improvement in {subject.name}: Average Score: {subject.average_score():.2f}")


def main():
    tasks = []
    subjects = []

    print("Welcome to the Student Task Manager and Performance Tracker!")

    # Phase 1: Task Management
    print("\nTask Management")
    while True:
        task = input_task()
        tasks.append(task)
        more_tasks = input("Would you like to add another task? (y/n): ").lower()
        if more_tasks != 'y':
            break

    display_tasks(tasks)

    # Phase 2: Performance Tracking
    print("\nPerformance Tracking")
    while True:
        subject_name = input("Enter subject name (or type 'done' to finish): ")
        if subject_name.lower() == 'done':
            break
        subject = Subject(subject_name)
        input_scores_for_subject(subject)
        subjects.append(subject)

    display_performance(subjects)

   

if __name__ == "__main__":
    main()
