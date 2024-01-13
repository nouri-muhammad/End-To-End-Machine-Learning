import sys


def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  
    # exc_tb will have the detail of in which file and on which line the error occured
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error [{str(error)}] occured in python script: [{file_name}] in line no.[{exc_tb.tb_lineno}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_details(error=error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
