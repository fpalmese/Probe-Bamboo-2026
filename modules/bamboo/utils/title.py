import cowsay
import os


def print_title():
    # Clear shell
    os.system("cls" if os.name == "nt" else "clear")

    my_fish = r"""
    \
    \  
            /`·.¸
        /¸...¸`:·
    ¸.·´  ¸   `·.¸.·´)
    : © ):´;      ¸  {
    `·.¸ `·  ¸.·´\`·¸)
        `\\´´\¸.·´
    """

    # Fun title
    cowsay.draw("Boosting some bits!", my_fish)
