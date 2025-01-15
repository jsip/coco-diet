import time

schedule = [
    {"id": 1, "type": "meal", "start_time": 8, "end_time": 10},
    {"id": 2, "type": "snack", "start_time": 12, "end_time": 13},
    {"id": 3, "type": "meal", "start_time": 19, "end_time": 21},
]


class Schedule:
    @staticmethod
    def current_hour():
        return int(time.strftime("%H"))

    @staticmethod
    def current_slot():
        current_hour = Schedule.current_hour()
        return next((slot for slot in schedule if slot["start_time"] <= current_hour < slot["end_time"]), None)

    @staticmethod
    def current_meal():
        current_slot = Schedule.current_slot()
        return current_slot["type"] if current_slot else None

    @staticmethod
    def can_eat():
        return Schedule.current_meal() in {"meal", "snack"}

    @staticmethod
    def is_meal():
        return Schedule.current_meal() == "meal"

    @staticmethod
    def is_snack():
        return Schedule.current_meal() == "snack"
