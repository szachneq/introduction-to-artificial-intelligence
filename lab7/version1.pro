% Define the days of the week
day(1, 'Monday').
day(2, 'Tuesday').
day(3, 'Wednesday').
day(4, 'Thursday').
day(5, 'Friday').
day(6, 'Saturday').
day(7, 'Sunday').

% Convert DDMM to day, month, and year
convert_ddmm_to_date(DDMM, date(Y, M, D)) :-
    atom_chars(DDMM, [D1, D2, M1, M2]),
    atom_concat(D1, D2, DayAtom),
    atom_concat(M1, M2, MonthAtom),
    atom_number(DayAtom, D),
    atom_number(MonthAtom, M),
    Y = 2024.

% Convert day, month, and year to DDMM
convert_date_to_ddmm(date(_, M, D), DDMM) :-
    format(atom(DDMM), '~|~`0t~d~2+~|~`0t~d~2+', [D, M]).

% Calculate the next working day
next_working_day(Date, NextDate) :-
    next_day(Date, NextDate1),
    (is_weekend(NextDate1) -> next_working_day(NextDate1, NextDate); NextDate = NextDate1).

% Check if a date is a weekend
is_weekend(date(Y, M, D)) :-
    day_of_the_week(date(Y, M, D), W),
    (W = 6; W = 7).

% Calculate the next day
next_day(date(Y, M, D), date(Y, M, NextD)) :-
    D < 28, NextD is D + 1.
next_day(date(Y, M, 28), date(Y, M, 29)) :- member(M, [1, 3, 5, 7, 8, 10, 12]).
next_day(date(Y, M, 28), date(Y, M, 29)) :- M \= 2.
next_day(date(Y, M, 29), date(Y, M, 30)) :- member(M, [1, 3, 5, 7, 8, 10, 12]).
next_day(date(Y, M, 29), date(Y, M, 30)) :- M \= 2.
next_day(date(Y, 2, 29), date(Y, 3, 1)).
next_day(date(Y, M, 30), date(Y, M, 31)) :- member(M, [1, 3, 5, 7, 8, 10, 12]).
next_day(date(Y, M, 30), date(Y, M, 31)) :- M \= 4, M \= 6, M \= 9, M \= 11.
next_day(date(Y, M, 31), date(Y, NextM, 1)) :- NextM is M + 1.
next_day(date(Y, 12, 31), date(NextY, 1, 1)) :- NextY is Y + 1.

% Calculate the resulting date after adding working days
resulting_date(StartDate, 0, StartDate).
resulting_date(StartDate, N, ResultDate) :-
    N > 0,
    next_working_day(StartDate, NextDate),
    N1 is N - 1,
    resulting_date(NextDate, N1, ResultDate).

% Main predicate to calculate the day and date after adding working days
n_work_days(DDMM, N, Result) :-
    convert_ddmm_to_date(DDMM, StartDate),
    resulting_date(StartDate, N, ResultDate),
    convert_date_to_ddmm(ResultDate, ResultDDMM),
    day_of_the_week(ResultDate, W),
    day(W, DayName),
    format(atom(Result), '~w, ~w', [DayName, ResultDDMM]).

% Helper function to get the day of the week using Zeller's Congruence
zellers_congruence(Y, M, D, DayOfWeek) :-
    (M = 1 -> Y1 is Y - 1, M1 is 13; M = 2 -> Y1 is Y - 1, M1 is 14; Y1 is Y, M1 is M),
    K is Y1 mod 100,
    J is Y1 // 100,
    F is D + ((13 * (M1 + 1)) // 5) + K + (K // 4) + (J // 4) - (2 * J),
    DayOfWeek is ((F + 5) mod 7) + 1.

% day_of_the_week predicate using Zeller's Congruence
day_of_the_week(date(Y, M, D), W) :-
    zellers_congruence(Y, M, D, W).

% Example query
% ?- n_work_days("2205", 6, Result).
% Result = "Tuesday, 3005".
