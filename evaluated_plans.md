## Bowl Over Cup code

On a table lies a bowl and a mug. To the right of the table lies the mug. The bowl has been placed over the mug. 

If you are commanding the robot, tell me step-by-step how to get the mug and place it to the left of the table.

```python
def fetch_mug():
    # Find the bowl and mug
    bowl_id = find("bowl", "over the mug")
    mug_id = find("mug", "under the bowl")

    # Get the locations of the bowl and mug
    bowl_location = get_location(bowl_id)
    mug_location = get_location(mug_id)

    # Move the robotic arm to the bowl's location and pick it up
    move_arm(bowl_location)
    pick(bowl_id)

    # Move the bowl to a safe location
    safe_location = [bowl_location[0], bowl_location[1] + 0.5, bowl_location[2]]
    place(bowl_id, safe_location)

    # Move the robotic arm to the mug's location and pick it up
    move_arm(mug_location)
    pick(mug_id)

    # Move the mug to the left side of the table
    left_of_table_location = [mug_location[0], mug_location[1] - 1.0, mug_location[2]]
    place(mug_id, left_of_table_location)

    # Signal the end of the program
    no_action()
```

## Bowls in a bin (stacking)