class ValidteJson():
    def __init__(self):
        self.except_obj_start = True
        self.except_key_or_close = False
        self.in_string = False
        self.except_colon = False
        self.except_value = False
        self.in_number = False
        self.except_comma_or_close = False

    def get_next_obj(self):
        if self.except_obj_start:
            self.except_obj_start = False
            self.except_key_or_close = True
            return "{"
        elif self.except_key_or_close :
            self.except_key_or_close = False
            self.in_string = True
            return '"'