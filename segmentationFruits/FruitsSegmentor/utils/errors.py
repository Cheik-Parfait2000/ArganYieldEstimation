

# ================== Custom exceptions :  errors.py ===============================
class InvalidFileError(Exception):
  """
  This exception is raised if the given file is not among the recognized files extensions like .jpg, .yml, etc

  message: explanation message
  """
  def __init__(self, message):
    self.message = message

    super().__init__(self.message)


class InvalidShapeError(Exception):
  """
  This exception is raised if the given image doesn't match the expected shape

  message: explanation message
  """
  def __init__(self, message):
    self.message = message

    super().__init__(self.message)


class InvalidTypeError(Exception):
  """
  This exception is raised if the type received is not what is expected

  message: explanation message
  """
  def __init__(self, message):
    self.message = message

    super().__init__(self.message)