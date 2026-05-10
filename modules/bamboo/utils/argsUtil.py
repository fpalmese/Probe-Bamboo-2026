from rich.panel import Panel

def argsHandler(parser, console) -> None:
    parser.add_argument("-M", type=int, help="number of iterations")
    parser.add_argument("-F", type=int, help="number of filters to use")
    parser.add_argument(
        "-X", type=int, help="number of head rows to use from the dataset"
    )
    parser.add_argument("-d", action="store_true", help="use debug dataset")
    parser.add_argument(
        "-rb", action="store_true", help="remove best filters at each iteration"
    )
    args = parser.parse_args()

    if args.M is None:
        console.print(
            Panel("[!] Argument M is missing! Setting it to 1.", style="bold red"),
            style="bold red",
        )
        args.M = 1

    if args.F is None:
        console.print(
            Panel("[!] Argument F is missing! Setting it to 16.", style="bold red"),
            style="bold red",
        )
        args.F = 16

    if args.M > args.F and args.F != 0:
        console.print(
            Panel(
                "[!] The number of iterations should be less than the number of filters.",
                style="bold red",
            ),
            style="bold red",
        )
    return args