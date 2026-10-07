# Account support guide

## Password resets

Users reset a password from the **Forgot password** link on the sign-in page. The
service sends a reset link to the verified account email address. A reset link
expires after 30 minutes and can be used only once. If the link has expired,
the user must request a new one rather than reusing an old email. Support may
confirm that the request was sent, but must not ask for a password or set one
on the user's behalf.

## Email-address changes

An authenticated user changes their contact email in **Profile > Email**. The
new address receives a verification message; the old address remains active
until that verification is completed. If the user no longer controls the old
address, the account owner must open a support request with their organization
administrator. Support escalates identity-review cases to the Account Recovery
queue and does not accept a screenshot as proof of ownership.

## Locked accounts

Five unsuccessful sign-in attempts lock an account for 15 minutes. The user
can wait for the lock to clear or use the password-reset procedure. An
organization administrator may unlock a member sooner from the Admin Console.
For repeated lockouts after a successful reset, send the account ID and the
time of the last attempt to the Account Recovery queue.
