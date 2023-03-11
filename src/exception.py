import sys

def error_msg_details(error,error_details:sys):
    _,_,exc_tb=error.details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msg="Error at {2} \n Error: {0} at line {1}".format(error_details,error_details.tb_lineno,file_name)
    return error_msg

class CustomException(Exception):
    def __init__(self,error_msg:str,error_details:sys):

        super().__init__(error_msg)
        self.error_msg=error_msg_details(error_msg,error_details)

    def __str__(self):
        return self.error_msg 