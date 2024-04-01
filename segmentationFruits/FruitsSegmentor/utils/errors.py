

# ================== Custom exceptions :  errors.py ===============================
class InvalidFileError(Exception):
  """
  This exception is raised if the given file is not among the recognized files:
  .pt/.yaml/.yml

  message: explanation message
  """
  def __init__(self, message):
    self.message = message

    super().__init__(self.message)