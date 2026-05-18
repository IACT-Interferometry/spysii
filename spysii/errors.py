class MixingTargetsError(Exception):
    def __init__(self, target1, target2):
        self.target1 = target1
        self.target2 = target2

    def __str__(self):
        return repr('While analysing runs of the target {},'.format(self.target1) +
                    'a run with a different target ({}) was encountered.'.format(self.target1))


class MissingTargetCoordinateError(Exception):
    def __init__(self, target_name):
        self.target_name = target_name

    def __str__(self):
        return repr('ERROR: Impossible to retrieve the sky coordinate of target {}.\n'.format(self.target_name) +
                    'This may be caused by the absence of internet connection (needed by astropy).\n')


class WrongDataFormat(Exception):
    def __init__(self, data_format):
        self.data_format = data_format

    def __str__(self):
        return repr('ERROR: The data format you specified ({}) has not been implemented.'.format(self.data_format))


class WrongPath(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr('ERROR: {}.'.format(self.message))


class WrongTimePeriod(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr('ERROR: {}.'.format(self.message))


class WrongTimeFormat(Exception):
    def __str__(self):
        return repr('ERROR: Could not identify the time provided. Please provide a Time instance, or a str in isot.')


class WrongDcReport(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr("ERROR: Seems like the dc report {} is either empty or not valid.".format(self.message))
