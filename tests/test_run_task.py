from scripts.run_task import build_arg_parser


def test_arg_parser_requires_task_positional():
    # Arrange
    parser = build_arg_parser()

    # Act
    args = parser.parse_args(["do something", "--headless", "--max-steps", "10"])

    # Assert
    assert args.task == "do something"
    assert args.headless is True
    assert args.max_steps == 10


def test_arg_parser_defaults_headless_false_and_max_steps_none():
    # Arrange
    parser = build_arg_parser()

    # Act
    args = parser.parse_args(["do something"])

    # Assert
    assert args.headless is False
    assert args.max_steps is None
