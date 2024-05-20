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
    %If the day is less than 28, return the next day 
    D < 28, NextD is D + 1.
% Handles the case where the date is the 28th day of a month with 31 days. (it returns 29th day)
next_day(date(Y, M, 28), date(Y, M, 29)) :- member(M, [1, 3, 5, 7, 8, 10, 12]).
% Handles the case where the date is the 28th day of any month except February.
% If the month M is not February (2), the next day after the 28th is the 29th
next_day(date(Y, M, 28), date(Y, M, 29)) :- M \= 2.
% Handles the case where the date is the 29th day of a month with 31 days.
% If the month M is one of January, March, May, July, August, October, or December, the next day after the 29th is the 30th.
next_day(date(Y, M, 29), date(Y, M, 30)) :- member(M, [1, 3, 5, 7, 8, 10, 12]).
% Handles the case where the date is the 29th day of any month except February.
% If the month M is not February (2), the next day after the 29th is the 30th.
next_day(date(Y, M, 29), date(Y, M, 30)) :- M \= 2.
% Handles the transition from February 29 to March 1.
% If the date is February 29, the next day is March 1.
next_day(date(Y, 2, 29), date(Y, 3, 1)).
% Handles the case where the date is the 30th day of a month with 31 days.
% If the month M is one of January, March, May, July, August, October, or December, the next day after the 30th is the 31st.
next_day(date(Y, M, 30), date(Y, M, 31)) :- member(M, [1, 3, 5, 7, 8, 10, 12]).
% Handles the transition from the 31st day of a month to the 1st day of the next month.
% Increments the month M by 1 to get NextM and sets the day to 1.
next_day(date(Y, M, 31), date(Y, NextM, 1)) :- NextM is M + 1.
% Handles the transition from December 31 to January 1 of the next year.
% Increments the year Y by 1 to get NextY and sets the month and day to January 1.
next_day(date(Y, 12, 31), date(NextY, 1, 1)) :- NextY is Y + 1.

% Calculate the resulting date after adding working days
resulting_date(StartDate, 0, StartDate).
resulting_date(StartDate, N, ResultDate) :-
    N > 0,
    next_working_day(StartDate, NextDate),
    N1 is N - 1,
    resulting_date(NextDate, N1, ResultDate).

day_of_the_week(Date, W) :-
    format_time(atom(Weekday), '%u', Date),
    atom_number(Weekday, W).

% Main predicate to calculate the day and date after adding working days
n_work_days(DDMM, N) :-
    convert_ddmm_to_date(DDMM, StartDate),
    resulting_date(StartDate, N, ResultDate),
    convert_date_to_ddmm(ResultDate, ResultDDMM),
    day_of_the_week(ResultDate, W),
    day(W, DayName),
    format(atom(Result), '~w, ~w', [DayName, ResultDDMM]),
    write(Result).


% Example query
% ?- n_work_days("2205", 6).
% "Tuesday, 3005".
% true
